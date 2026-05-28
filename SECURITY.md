# Security

DR-OPIC verifies generated Python by writing a temporary file and running it in a
subprocess with a timeout and minimal environment. This is not a security
sandbox.

For untrusted model output:

- run verification in a container, VM, or managed sandbox
- disable network access
- mount only disposable directories
- do not expose production secrets
- set strict CPU, memory, and wall-clock limits
- review tool permissions before agentic use

Please report security issues privately through the repository owner instead of
opening a public issue with exploit details.
