# The 6502: Architecture, Memory Layout, and the Little Old Lady's Syntax

The 6502 is the little old lady of microprocessors. She arrived in 1975,
cost twenty-five dollars, and never left. She ran the Apple II, the
Commodore 64, the NES, and the Atari 2600. She is eight bits wide, one
megahertz slow, and still the clearest teacher of how a computer
actually works — because there is nothing to hide behind. No cache, no
pipeline, no multiply, no floating point. Just a handful of registers,
sixty-four kilobytes of address space, and a syntax so terse it reads
like shorthand.

## The architecture

The 6502 has six registers. The accumulator (A) is where arithmetic
happens. Two index registers (X and Y) count, offset, and loop. The stack
pointer (SP) tracks a 256-byte stack. The program counter (PC) holds the
address of the next instruction. And the status register (P) holds seven
flags: Negative, Overflow, Break, Decimal, Interrupt-disable, Zero, and
Carry. Every instruction that touches a value sets the flags, and every
branch reads them.

There are fifty-six instructions and thirteen addressing modes. The
instruction set is small enough to memorize. Loads and stores move bytes.
`ADC` and `SBC` add and subtract (with the carry flag as the ninth bit).
`AND`, `ORA`, `EOR` do bitwise logic. `ASL`, `LSR`, `ROL`, `ROR` shift
and rotate. `CMP` compares. `JMP`, `JSR`, and `RTS` jump, call, and
return. `BEQ`, `BNE`, `BMI`, `BPL`, `BCC`, `BCS`, `BVS`, `BVC` branch on
the flags. `INX`, `INY`, `DEX`, `DEY` count. The rest is addressing.

What the 6502 does *not* have is as important as what it does. There is
no multiply instruction. There is no divide. There is no 16-bit
arithmetic (you build it from 8-bit adds with carry). There is no
floating point, no vector unit, and no memory protection. If you want
a multiply, you add the same number over and over, or you shift and
add. That is why ternary weights fit this chip so well: a weight of
1 means add, −1 means subtract, 0 means skip. No multiply needed.

The chip runs at about one megahertz on the machines it powered. Each
instruction takes two to seven cycles. Everything is deterministic.
There are no caches to miss, no branch predictors to fool, no
speculative execution. The cost of a program is the sum of its
instructions, and you can count them by hand.

## The memory layout

The 6502 sees 64 kilobytes, addressed `$0000` to `$FFFF`, and every
address means something. The layout is not an abstraction — it is the
machine's anatomy.

Zero page (`$0000–$00FF`) is the first 256 bytes. It is special because
the 6502 can address it with a single byte instead of two, so zero-page
instructions are shorter and faster. Programmers keep their hottest
variables here: pointers, counters, accumulators. In this repo, the
ternary kernel's pointers live in zero page (`InputPtr` at `$00`,
`WeightPtr` at `$02`, the accumulator at `$04–$05`, the length at `$06`),
because the kernel runs thousands of times and every cycle counts.

The stack (`$0100–$01FF`) is the second page. The stack pointer starts
at `$FF` and grows downward. `JSR` pushes a return address; `RTS`
pulls it back. `PHA` and `PLA` push and pull the accumulator. The stack
is only 256 bytes, and there is no protection.

RAM (`$0200` onward) is working memory. On the classic machines it was
two kilobytes (`$0000–$07FF`), sometimes more with bank switching.
This repo puts the transformer's activations and gradients in RAM
starting at `$0200`: the input vectors, Q, K, V, scores, attention
weights, attention output, FFN hidden layer, residual, logits — then
every gradient, from `dLogits` back through `dEmb`. The weight tables
(ternary, in RAM so training can update them) live at `$0500` and up;
the latent weights (the higher-precision values SGD actually moves)
live at `$0400` and up. The split is deliberate: the ternary tables are
what the forward pass reads, the latents are what the backward pass
writes, and the quantizer bridges them.

Above RAM sit the machine's I/O and ROM. The top of memory
(`$FFFA–$FFFF`) holds the interrupt vectors: NMI, RESET, IRQ. The RESET
vector is where the machine starts — it points at the entry point of
the program. This repo's `main.asm` places its vectors there, so the
transformer is the machine's whole job.

## The little old lady's syntax

The 6502's assembly syntax is the little old lady's voice: terse,
practical, and completely without ornament. Every instruction is a
three-letter mnemonic. Every operand says exactly how it is addressed,
in a handful of forms that have not changed since 1975.

An immediate value wears a `#`: `LDA #0` loads the number zero.
A zero-page address stands bare: `LDA $06` loads the byte at `$06`.
An absolute address is two bytes: `LDA $0200` loads from RAM.
An indexed address adds X or Y: `LDA $0200,X` loads from `$0200 + X`.
An indirect address wraps in parentheses: `LDA ($00),Y` loads from the
address stored at `$00–$01`, plus Y. That last form — `(pointer),Y` —
is the workhorse of the ternary kernel: it walks a vector one element
at a time, `Y` counting from zero to the length.

Branches are two letters and a label: `BEQ Done`, `BMI Subtract`,
`BCC NextElement`. They read the flags the previous instruction set.
There are no `if` statements, no `for` loops, no function calls with
arguments. There is `JSR` (jump to subroutine) and `RTS` (return).
Arguments travel in registers, in zero page, or in fixed memory. The
calling convention is yours to invent — the machine does not check.

This is why the syntax feels historical: it *is* historical. It comes
from an era when programmers wrote machine code by hand, counted
cycles, and knew the address of every byte they used. The mnemonics
are abbreviations of English verbs — load, store, add, subtract,
compare, jump, branch — because the people who designed the chip
wanted the code to read like instructions to a clerk.

The little old lady's syntax has survived because it is honest. It
does not pretend the machine is something it is not. When you write
`LDA (WeightPtr),Y` / `BEQ NextElement`, you are saying: load the
weight, and if it is zero, skip the math. The sparsity is not a
library feature. It is a branch. The transformer is not a framework.
It is a loop.
