"""A hierarchical timing wheel, one per reactor.

Every connection needs a deadline. Idle timeouts, header read timeouts, SSE
heartbeats, and later the request budget the scheduler enforces. With a
thousand connections that is a thousand deadlines that mostly never fire,
because a connection almost always does something before its timer does.

The obvious implementations both get this wrong in the same direction. One
timer descriptor per connection doubles the descriptor count and puts every
deadline through the poller, which is a syscall per arm and a syscall per
cancel. A sorted list or a heap makes arming O(log n) and, worse, makes the
common operation, pushing a deadline further out because the connection was
active, cost the same as creating it.

A timing wheel makes arming, cancelling and firing O(1), because a deadline is
just an index. Time is chopped into ticks, the ticks into slots, and a timer
goes into the slot its deadline lands in. Advancing means walking one slot and
firing what is in it. Four levels of sixty four slots cover deadlines from one
tick to nineteen days, and the levels cascade: a timer far in the future sits in
a coarse slot until time gets close enough for it to move down to a fine one.
The cascade is the only interesting part, and it is nine lines.

The tick is 100 ms. That is the granularity of every deadline here, and it is
chosen against what the deadlines are for. An idle timeout of thirty seconds
does not care about 100 ms, an SSE heartbeat of fifteen seconds does not care,
and a finer tick would mean the reactor waking more often to do nothing, which
is the thing that shows up on a laptop's battery and in a container's CPU
share.

Cancelling is lazy and the reason is worth a sentence. A cancelled timer is
marked dead and left where it is, freed when its slot is next walked, because
unlinking from a singly linked slot list would mean either a doubly linked list
or a scan. Nothing here creates enough dead timers for that to matter: the one
operation that would, refreshing an idle deadline on every read, is not done by
cancelling and rearming. The reactor lets the timer fire, checks how long the
connection has actually been idle, and rearms only if it has more to wait.
"""

comptime TICK_MS = 100
"""Resolution. Every deadline is rounded up to a multiple of this."""

comptime SLOT_BITS = 6
comptime SLOTS = 64
comptime SLOT_MASK = 63
comptime LEVELS = 4
"""Level 0 covers 6.4 seconds, level 1 nearly seven minutes, level 2 seven
hours, level 3 nineteen days. Anything past that is clamped, and nothing molla
does has a deadline in nineteen days."""

comptime MAX_TICKS_PER_ADVANCE = 65536
"""If the reactor was blocked for longer than this many ticks, an hour and
fifty minutes, the wheel jumps rather than walking every slot in between. That
only happens if the process was stopped, and firing every timer at once after a
SIGCONT is better than spending a second in the wheel."""

comptime NO_TIMER = -1


struct Timer(Copyable, ImplicitlyCopyable, Movable):
    """One deadline. `token` is whatever the caller wants back when it fires,
    which for the reactor is a connection slot."""

    var token: Int
    var deadline: Int
    """Absolute tick, counted from when the wheel was made."""

    var next: Int
    """The next timer in the same slot, or -1. Slots are singly linked through
    the slab so a timer costs no allocation of its own."""

    var live: Bool

    def __init__(out self):
        self.token = 0
        self.deadline = 0
        self.next = NO_TIMER
        self.live = False


struct Wheel(Movable):
    """Deadlines for one reactor. Not thread safe, like everything else a
    reactor owns."""

    var slots: List[Int]
    """LEVELS times SLOTS heads, -1 for an empty slot."""

    var timers: List[Timer]
    """The slab. Ids are indices into this and stay valid until released."""

    var free_head: Int
    var current: Int
    """Ticks since the wheel was made."""

    var start_ms: Int
    var pending: Int
    """Live timers. The reactor uses this to decide whether it can block
    indefinitely in the poller."""

    def __init__(out self, now_ms: Int):
        self.slots = List[Int](length=LEVELS * SLOTS, fill=NO_TIMER)
        self.timers = List[Timer]()
        self.free_head = NO_TIMER
        self.current = 0
        self.start_ms = now_ms
        self.pending = 0

    def _acquire(mut self) -> Int:
        """A slab entry, reused if one is free."""
        if self.free_head != NO_TIMER:
            var id = self.free_head
            self.free_head = self.timers[id].next
            return id
        self.timers.append(Timer())
        return len(self.timers) - 1

    def _release(mut self, id: Int):
        self.timers[id].live = False
        self.timers[id].next = self.free_head
        self.free_head = id

    def _slot_for(self, deadline: Int) -> Int:
        """Which of the 256 slots a deadline belongs in right now.

        The level is chosen by how far away the deadline is, and the slot
        inside the level by the bits of the deadline that level indexes. A
        deadline more than a level can express sits one level up until the
        cascade brings it down."""
        var delta = deadline - self.current
        if delta < 1:
            # Overdue, which happens when a cascade brings something down at
            # exactly its deadline. Into the slot being walked right now, so it
            # fires on this tick instead of waiting a full revolution.
            return self.current & SLOT_MASK
        var level = 0
        var span = SLOTS
        while level < LEVELS - 1 and delta >= span:
            level += 1
            span = span << SLOT_BITS
        var index = (deadline >> (SLOT_BITS * level)) & SLOT_MASK
        return level * SLOTS + index

    def _link(mut self, id: Int):
        var slot = self._slot_for(self.timers[id].deadline)
        self.timers[id].next = self.slots[slot]
        self.slots[slot] = id

    def add(mut self, token: Int, delay_ms: Int) -> Int:
        """Arm a timer. Returns its id, which is what `cancel` takes.

        A delay of zero or less still waits one tick. Firing inside the call
        that armed it would mean a caller could be re-entered from its own
        arming path, and no deadline in a server is worth that."""
        var ticks = (delay_ms + TICK_MS - 1) // TICK_MS
        if ticks < 1:
            ticks = 1
        var id = self._acquire()
        self.timers[id].token = token
        self.timers[id].deadline = self.current + ticks
        self.timers[id].live = True
        self._link(id)
        self.pending += 1
        return id

    def cancel(mut self, id: Int):
        """Mark a timer dead. It is freed when its slot is next walked.

        Safe to call on a timer that already fired, because firing frees the
        slab entry and a freed entry is not live, so this is a no op rather
        than a corruption. That matters: the reactor cancels the idle timer of
        every connection it closes, including the ones it is closing because
        that timer fired."""
        if id < 0 or id >= len(self.timers):
            return
        if not self.timers[id].live:
            return
        self.timers[id].live = False
        self.pending -= 1

    def deadline_of(self, id: Int) -> Int:
        """The absolute tick a timer is set for. For tests."""
        if id < 0 or id >= len(self.timers):
            return -1
        return self.timers[id].deadline

    def _cascade(mut self, level: Int, index: Int):
        """Move everything in one coarse slot down to where it belongs now.

        This is the whole trick. A timer an hour out sits in a level 2 slot and
        is never looked at until the hour is nearly up, at which point one pass
        moves it to a level 1 slot, and later to a level 0 one. Each timer
        moves at most once per level, so the amortised cost of a deadline is
        constant no matter how far out it was set."""
        var slot = level * SLOTS + index
        var id = self.slots[slot]
        self.slots[slot] = NO_TIMER
        while id != NO_TIMER:
            var next = self.timers[id].next
            if self.timers[id].live:
                self._link(id)
            else:
                self._release(id)
            id = next

    def _run_slot(mut self, index: Int, mut fired: List[Int]):
        """Fire what is due in a level 0 slot, keep what is not.

        A timer can be in this slot with a deadline still in the future when it
        was cascaded down from a coarser level mid tick, so the deadline is
        checked rather than assumed."""
        var id = self.slots[index]
        self.slots[index] = NO_TIMER
        while id != NO_TIMER:
            var next = self.timers[id].next
            if not self.timers[id].live:
                self._release(id)
            elif self.timers[id].deadline <= self.current:
                fired.append(self.timers[id].token)
                self.pending -= 1
                self._release(id)
            else:
                self._link(id)
            id = next

    def _tick(mut self, mut fired: List[Int]):
        self.current += 1
        var index = self.current & SLOT_MASK
        if index == 0:
            # A level only cascades when the level below it has wrapped, which
            # is what makes each timer move once per level rather than once per
            # tick.
            var level = 1
            while level < LEVELS:
                var at = (self.current >> (SLOT_BITS * level)) & SLOT_MASK
                self._cascade(level, at)
                if at != 0:
                    break
                level += 1
        self._run_slot(index, fired)

    def advance(mut self, now_ms: Int, mut fired: List[Int]) -> Int:
        """Walk the wheel up to now and collect the tokens that came due.

        Returns how many fired. `fired` is cleared first, so the caller can
        keep one list for the life of the reactor and never allocate on this
        path."""
        fired.clear()
        var target = (now_ms - self.start_ms) // TICK_MS
        if target > self.current + MAX_TICKS_PER_ADVANCE:
            target = self.current + MAX_TICKS_PER_ADVANCE
        while self.current < target:
            self._tick(fired)
        return len(fired)

    def next_timeout_ms(self) -> Int:
        """How long the reactor may sit in the poller.

        One tick when anything is armed, so no deadline is late by more than
        the resolution, and -1 when nothing is, so an idle server with no
        connections does not wake ten times a second to look at empty slots."""
        if self.pending == 0:
            return -1
        return TICK_MS

    def now_tick(self) -> Int:
        return self.current
