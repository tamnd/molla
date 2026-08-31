"""Connection handling on top of the syscall layer.

Nothing here knows what a request is. HTTP arrives in M1 and sits above this,
reading and writing through the same event loop shape the echo spike proves out.
"""
