# Security Policy

## Supported Versions

MetaXuda is an experimental project. Security fixes are provided for the latest published release and the current `main` branch.

| Version | Supported |
| --- | --- |
| 2.0.x | Yes |
| 1.0.x | No |

## Reporting a Vulnerability

Please do not open a public issue for a suspected vulnerability.

Use GitHub's private vulnerability reporting feature from the repository's **Security** tab. Include:

- The affected version or commit
- A clear description of the issue
- Steps to reproduce it
- The potential impact
- Any suggested mitigation

Reports will be reviewed as soon as practical. Confirmed issues will be addressed privately before public disclosure when possible.

## Scope

Security reports may include unsafe memory behavior, unintended code execution, library-loading vulnerabilities, credential exposure, or flaws in the CUDA-to-Metal runtime boundary.

MetaXuda is a research prototype and does not provide the same security guarantees as production CUDA runtimes. Avoid using it with untrusted kernels, inputs, or binary artifacts.
