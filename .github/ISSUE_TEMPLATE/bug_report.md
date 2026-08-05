---
name: Bug report
about: Report a problem with MOAD (incorrect output, crash, or unexpected behavior).
title: "[Bug] "
labels: bug
---

Please paste text, code, and numbers instead of screenshots when
possible. A small reproducible example is more useful than a long
description. For long error messages, please use a fenced code block
(triple backticks) so the traceback survives Markdown formatting.

## What happened?

Describe the problem in 1-3 sentences.

## What were you trying to do?

Example: compute XAS, build a basis, compare with Quanty, run a benchmark, etc.

## Code or command

The smallest code/command that shows the issue (a few lines that
someone else can copy, paste, and run). If it depends on a data file,
mention which one.

```julia
using MOAD

# code here
```

## Output / error

Paste the full error message or unexpected number.

```
output here
```

## Expected result

What did you expect instead? If this is a numerical disagreement,
include the reference (paper, Quanty output, hand calculation, etc.).

## Environment

- MOAD commit / version: (paste output of `Pkg.status("MOAD")`)
- Julia version: (paste output of `versioninfo()` first line)
- OS:
