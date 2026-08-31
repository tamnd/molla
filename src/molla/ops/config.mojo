"""Settings, and where each one came from.

Four places a value can come from, in this order of strength: a flag on the
command line, an environment variable, a file, and the default compiled in
here. That order is not a preference, it is the only one that makes a container
usable. The image ships a file, the orchestrator sets environment variables per
deployment, and a person debugging at three in the morning overrides one thing
on the command line without editing either.

The part that is unusual is that every setting remembers which of the four it
came from, and `molla config get` prints it. Most of the time somebody is
reading a config value they are not asking what it is, they already think they
know what it is. They are asking why it is that, and an effective value with no
provenance answers the wrong question. It costs one integer per setting.

There is no global. `Config` is built once at startup and passed down, the same
as `ServerContext`, and a test that wants a different configuration makes
another one. Two configs in one process is a normal thing here rather than a
thing to work around.

The file format is `key = value`, one per line, `#` starts a comment, and blank
lines are ignored. Not TOML and not YAML. molla needs about a dozen scalars and
a parser for either of those is a week of work and a supply of parsing bugs, so
this reads the format it actually needs. The day a setting needs to be a table,
that is the day to reconsider, and it is not this day.
"""

from std.os.env import getenv

from molla.sys.file import FileInfo, close_fd, fstat, open_read, pread_all

comptime FROM_DEFAULT = 0
comptime FROM_FILE = 1
comptime FROM_ENV = 2
comptime FROM_FLAG = 3
"""Ascending strength. `_apply` refuses a value from a weaker source than the
one already there, so the loader can read the four in any order and does not
have to be read backwards to know what wins."""

comptime MAX_FILE_BYTES = 65536
"""A config file bigger than this is not a config file. Refusing it is cheaper
than reading an arbitrary amount of whatever was pointed at."""

comptime CONFIG_ENV = "MOLLA_CONFIG"
"""Where to look for the file, if not the default path."""

comptime DEFAULT_CONFIG_PATH = "molla.conf"


def source_name(source: Int) -> StaticString:
    if source == FROM_FLAG:
        return "flag"
    if source == FROM_ENV:
        return "env"
    if source == FROM_FILE:
        return "file"
    if source == FROM_DEFAULT:
        return "default"
    return "unknown"


struct Setting(Copyable, ImplicitlyCopyable, Movable):
    """One setting, its value as text, and where the value came from.

    Text rather than a union, because everything here arrives as text from at
    least two of the four sources and a config file cannot say what type it
    meant anyway. Callers ask for `int_value` and get the default back if the
    text is not a number, which is checked at startup by `problems` rather than
    discovered on the request path.
    """

    var name: String
    var value: String
    var source: Int
    var fallback: String
    """The compiled in default, kept so `molla config get` can show what a
    setting would be without the file and the environment."""

    var help: String

    def __init__(
        out self, name: String, value: String, help: String = String("")
    ):
        self.name = name
        self.value = value
        self.source = FROM_DEFAULT
        self.fallback = value
        self.help = help

    def int_value(self, if_bad: Int = 0) -> Int:
        try:
            return Int(self.value)
        except:
            return if_bad

    def bool_value(self) -> Bool:
        """True, yes, on and 1 are true. Everything else is false.

        Deliberately not case sensitive and deliberately narrow. A config file
        that says `metrics = enabled` should be a problem somebody is told
        about at startup, not a quietly disabled metrics endpoint.
        """
        var v = self.value.lower()
        return v == "true" or v == "yes" or v == "on" or v == "1"

    def is_int(self) -> Bool:
        try:
            _ = Int(self.value)
            return True
        except:
            return False


comptime KEY_WORKERS = "workers"
comptime KEY_IDLE_MS = "idle_timeout_ms"
comptime KEY_DRAIN_MS = "drain_deadline_ms"
comptime KEY_LOG_LEVEL = "log_level"
comptime KEY_LOG_QUEUE = "log_queue_bytes"
comptime KEY_ADMIN = "admin_routes"
comptime KEY_METRICS = "metrics"


def _defaults() -> List[Setting]:
    """Every setting molla has, with the value it takes when nobody says.

    One list, in one place, because the alternative is a default written into
    whichever function first needed it and a second copy of it in the usage
    text. Adding a setting means adding a line here and nothing else.
    """
    var out = List[Setting]()
    out.append(
        Setting(
            KEY_WORKERS,
            "0",
            "I/O threads, or 0 for one per core within the cap",
        )
    )
    out.append(
        Setting(
            KEY_IDLE_MS,
            "60000",
            "how long a connection may sit idle before it is closed",
        )
    )
    out.append(
        Setting(
            KEY_DRAIN_MS,
            "10000",
            "how long a graceful shutdown waits before it cuts connections",
        )
    )
    out.append(
        Setting(KEY_LOG_LEVEL, "info", "debug, info, warn, error, or off")
    )
    out.append(
        Setting(
            KEY_LOG_QUEUE,
            "65536",
            "bytes of log ring per worker, rounded up to a power of two",
        )
    )
    out.append(Setting(KEY_ADMIN, "true", "serve the /molla routes"))
    out.append(
        Setting(KEY_METRICS, "true", "collect and export Prometheus metrics")
    )
    return out^


def _as_int(text: String) -> Int:
    """A number, or zero. Used for the compiled in defaults, which are all
    numbers, so the zero only exists because the compiler wants it to."""
    try:
        return Int(text)
    except:
        return 0


def env_name(key: String) -> String:
    """`idle_timeout_ms` becomes `MOLLA_IDLE_TIMEOUT_MS`.

    Derived rather than listed, so a setting cannot end up with an environment
    variable that does not match its name, which is the kind of thing nobody
    finds until it matters.
    """
    return String("MOLLA_") + key.upper()


struct Config(Movable):
    """Every setting, resolved, with its provenance."""

    var settings: List[Setting]
    var path: String
    """The file that was read, or empty if none was."""

    var complaints: List[String]
    """Things wrong with what the caller supplied. Collected rather than
    raised, because a person who has three typos in a config file should be
    told about all three at once and not one per restart."""

    def __init__(out self):
        self.settings = _defaults()
        self.path = String("")
        self.complaints = List[String]()

    def index_of(self, name: StringSpan) -> Int:
        for i in range(len(self.settings)):
            if self.settings[i].name == name:
                return i
        return -1

    def get(self, name: StringSpan) -> Setting:
        """The setting, or an empty one named for what was asked for.

        An unknown name is not an error here. `molla config get nonsense` says
        the name is unknown, which is the caller's problem, and every internal
        caller uses the `KEY_` constants and cannot get it wrong.
        """
        var at = self.index_of(name)
        if at < 0:
            return Setting(String(name), String(""))
        return self.settings[at]

    def value(self, name: StringSpan) -> String:
        return self.get(name).value

    def int_value(self, name: StringSpan) -> Int:
        """The number, or the compiled in default if what is there is not one.

        Falling back rather than raising, because `problems` has already been
        called at startup and refused to run with a setting that is not a
        number. A caller reaching here with bad text is a test or a caller that
        skipped the check, and a default is a better answer than a crash on the
        request path.
        """
        var at = self.index_of(name)
        if at < 0:
            return 0
        return self.settings[at].int_value(_as_int(self.settings[at].fallback))

    def bool_value(self, name: StringSpan) -> Bool:
        return self.get(name).bool_value()

    def _apply(mut self, name: StringSpan, value: StringSpan, source: Int):
        """Take a value if it comes from at least as strong a source.

        At least as strong rather than stronger, so a file that names the same
        key twice keeps the last one, which is what every config file in the
        world does.
        """
        var at = self.index_of(name)
        if at < 0:
            self.complaints.append(
                String("unknown setting '")
                + String(name)
                + "' from the "
                + String(source_name(source))
            )
            return
        if source < self.settings[at].source:
            return
        self.settings[at].value = String(value)
        self.settings[at].source = source

    def load_file(mut self, path: StringSpan) -> Bool:
        """Read `key = value` lines. False if the file could not be read.

        A missing file is not a complaint when it is the default path, since
        running with no config file is the normal case. A missing file that
        somebody named explicitly is, and that distinction is made by the
        caller rather than here.
        """
        var text = _read_file(path)
        if text.byte_length() == 0 and not _exists(path):
            return False
        self.path = String(path)
        var line_no = 0
        for line in text.split("\n"):
            line_no += 1
            var trimmed = String(line).strip()
            if trimmed.byte_length() == 0 or trimmed.startswith("#"):
                continue
            var eq = trimmed.find("=")
            if eq < 0:
                self.complaints.append(
                    String(path)
                    + " line "
                    + String(line_no)
                    + ": expected key = value"
                )
                continue
            var key = trimmed[byte=0:eq].strip()
            var val = trimmed[byte = eq + 1 : trimmed.byte_length()].strip()
            self._apply(key, val, FROM_FILE)
        return True

    def load_env(mut self):
        """Look for `MOLLA_<KEY>` for every setting molla knows about.

        Pull rather than push, because there is no portable way to walk the
        environment in Mojo 1.0 and because scanning for a prefix would mean a
        typo in an environment variable silently does nothing. This way a
        misspelled variable also silently does nothing, but the set of names
        that work is exactly the set `molla config get` prints, which is at
        least discoverable.
        """
        for i in range(len(self.settings)):
            var key = self.settings[i].name
            var found = getenv(env_name(key))
            if found.byte_length() == 0:
                continue
            self._apply(key, found, FROM_ENV)

    def load_flag(mut self, arg: StringSpan) -> Bool:
        """Take one `--key=value` argument. False if it is not one.

        Only the long form with an equals sign. A space separated form means
        the parser has to know which flags take a value, which means a table
        that goes stale, and `--workers 4` versus `--workers=4` is not a
        hardship worth a class of bug over.
        """
        var text = String(arg)
        if not text.startswith("--"):
            return False
        var body = text[byte = 2 : text.byte_length()]
        var eq = body.find("=")
        if eq < 0:
            return False
        var key = body[byte=0:eq].strip()
        var val = body[byte = eq + 1 : body.byte_length()].strip()
        if self.index_of(key) < 0:
            return False
        self._apply(key, val, FROM_FLAG)
        return True

    def problems(self) -> List[String]:
        """Everything wrong, checked once at startup.

        The type checks live here rather than in `int_value` because a setting
        that is not a number should stop the process with a message naming it,
        not silently become a default on the first request that reads it.
        """
        var out = List[String]()
        for i in range(len(self.complaints)):
            out.append(self.complaints[i])
        for i in range(len(self.settings)):
            var s = self.settings[i]
            if s.name == KEY_LOG_LEVEL:
                if level_of(s.value) < 0:
                    out.append(
                        String("log_level '")
                        + s.value
                        + "' is not one of debug, info, warn, error, off"
                    )
                continue
            if s.name == KEY_ADMIN or s.name == KEY_METRICS:
                continue
            if not s.is_int():
                out.append(
                    String(s.name)
                    + " should be a number and is '"
                    + s.value
                    + "', from the "
                    + String(source_name(s.source))
                )
        return out^

    def describe(self) -> String:
        """Every setting, its value, and where it came from. What `molla config
        get` prints when it is not given a name."""
        var out = String("")
        if self.path.byte_length() > 0:
            out += "file " + self.path + "\n"
        for i in range(len(self.settings)):
            out += describe_setting(self.settings[i]) + "\n"
        return out^


def describe_setting(setting: Setting) -> String:
    """One line: name, value, source, and the default when it is not the source.

    The default is shown only when something overrode it, because a line that
    says `workers = 0 (default, default 0)` is noise and a line that says
    `workers = 4 (env, default 0)` is the answer to the question that was asked.
    """
    var out = setting.name + " = " + setting.value
    out += " (" + String(source_name(setting.source))
    if setting.source != FROM_DEFAULT:
        out += ", default " + setting.fallback
    out += ")"
    return out^


comptime LEVEL_DEBUG = 0
comptime LEVEL_INFO = 1
comptime LEVEL_WARN = 2
comptime LEVEL_ERROR = 3
comptime LEVEL_OFF = 4
"""Levels live here rather than in `log.mojo` so that `Config.problems` can
check one without importing the logger, which would import the whole ring."""


def level_of(name: StringSpan) -> Int:
    """The level a name means, or -1 for a name that means nothing."""
    var n = String(name).lower()
    if n == "debug":
        return LEVEL_DEBUG
    if n == "info":
        return LEVEL_INFO
    if n == "warn" or n == "warning":
        return LEVEL_WARN
    if n == "error":
        return LEVEL_ERROR
    if n == "off" or n == "none":
        return LEVEL_OFF
    return -1


def level_name(level: Int) -> StaticString:
    if level <= LEVEL_DEBUG:
        return "debug"
    if level == LEVEL_INFO:
        return "info"
    if level == LEVEL_WARN:
        return "warn"
    if level == LEVEL_ERROR:
        return "error"
    return "off"


def load_config(args: List[String]) -> Config:
    """Build the configuration the way a process should: file, then
    environment, then flags.

    Read in ascending order of strength, which reads the way the precedence
    does. `_apply` would allow any order, and doing it in the order somebody
    reading this expects is worth more than the flexibility.
    """
    var config = Config()

    var path = getenv(CONFIG_ENV)
    var named = path.byte_length() > 0
    for i in range(len(args)):
        if args[i].startswith("--config="):
            path = String(args[i][byte = 9 : args[i].byte_length()])
            named = True
    if path.byte_length() == 0:
        path = String(DEFAULT_CONFIG_PATH)
    if not config.load_file(path) and named:
        config.complaints.append(
            String("could not read the config file '") + path + "'"
        )

    config.load_env()

    for i in range(len(args)):
        if args[i].startswith("--config="):
            continue
        _ = config.load_flag(args[i])

    return config^


def _exists(path: StringSpan) -> Bool:
    var opened = open_read(path)
    if not opened.is_ok():
        return False
    _ = close_fd(opened.value)
    return True


def _read_file(path: StringSpan) -> String:
    """The whole file as text, or empty when there is no file to read.

    Whole rather than streamed because it is a config file. Anything over
    `MAX_FILE_BYTES` is refused rather than truncated, since a half read config
    file is worse than no config file.
    """
    var opened = open_read(path)
    if not opened.is_ok():
        return String("")
    var fd = opened.value
    var info = FileInfo()
    var sized = fstat(fd, info)
    if not sized.is_ok() or info.size <= 0 or info.size > MAX_FILE_BYTES:
        _ = close_fd(fd)
        return String("")
    var size = info.size
    var raw = List[UInt8](capacity=size + 1)
    for _ in range(size):
        raw.append(0)
    var read = pread_all(fd, raw.unsafe_ptr(), size, 0)
    _ = close_fd(fd)
    if not read.is_ok():
        return String("")
    return String(StringSlice(unsafe_from_utf8=Span(raw)))
