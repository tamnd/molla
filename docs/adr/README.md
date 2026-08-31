# Decision records

`docs/design.md` holds the decisions molla is built on, D1 through D12, in the form of what they are and why. This directory holds the ones that were decided against measurements, at a gate, with the numbers written down next to the call.

The two are not redundant. `design.md` says what molla does today and stays short enough to read in one sitting. A record here says what was on the table at the time, what was measured, what got decided, and what would change the answer. When a decision is revisited, the design document is edited and the record is not, because a record that gets edited afterwards is worth nothing.

One file per decision, numbered in the order they were taken. A record never changes after it is accepted except to be marked superseded, with a link to whichever record replaced it.

| Record | Decision | Status |
| --- | --- | --- |
| [0001](0001-network-edge-stays-in-mojo.md) | The network edge stays in Mojo, D1 holds | Accepted, 2026-08-31 |
