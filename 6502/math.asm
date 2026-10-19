; ==================================================================
; 6502 TRANSFORMER — MATH HELPERS
; ca65 syntax.
; Conventions: 8-bit values are signed two's complement in A / memory.
; 16-bit values are little-endian (Lo first, Hi second).
; All routines are UNVERIFIED by execution (no 6502 emulator on this
; VM). They are written carefully, reviewed, and structured to be
; assembled with ca65 and run on a 6502 / emulator.
; ==================================================================
; Zero-page scratch used by this file (do not overlap $00-$06, the
; ternary kernel's registers):
;   $07-$08  MATH_PTR   general pointer
;   $09-$0A  MATH_PTR2  second pointer
;   $0B-$0C  MATH_T16   16-bit temp / MUL8 product (Lo, Hi)
;   $0D      MATH_T8    8-bit temp
;   $0E      MATH_SIGN  sign flag for multiply (0 = +, 1 = -)
;   $0F      MATH_CNT   loop counter / length
;   $10-$11  DOT_SUM    DOT8 running sum (Lo, Hi)
;   $12      DOT_IDX    DOT8 element index
; ==================================================================

MATH_PTR    = $07
MATH_PTR2   = $09
MATH_T16    = $0B
MATH_T16_HI = $0C
MATH_T8     = $0D
MATH_SIGN   = $0E
MATH_CNT    = $0F
DOT_SUM     = $10
DOT_SUM_HI  = $11
DOT_IDX     = $12

; ------------------------------------------------------------------
; MUL8 — signed 8-bit x signed 8-bit -> signed 16-bit.
;   In:  A = multiplicand, Y = multiplier
;   Out: MATH_T16 (Lo), MATH_T16_HI (Hi). A, X, Y clobbered.
;   Method: |A| added |B| times (repeated addition), sign fixup.
;   Our values are small (activations in [-140,140]), so at most
;   140 adds. Slow but obviously correct.
; ------------------------------------------------------------------
.proc MUL8
    STA MATH_T8            ; multiplicand
    STY MATH_T16           ; multiplier (temp)
    LDA #0
    STA MATH_SIGN
    STA MATH_T16_HI        ; product accumulator (low byte of product)
    ; |multiplicand|
    LDA MATH_T8
    BPL Mul8_PosA
    LDA #1
    STA MATH_SIGN
    LDA MATH_T8
    EOR #$FF
    CLC
    ADC #1
    STA MATH_T8            ; MATH_T8 = |A|
Mul8_PosA:
    ; |multiplier| -> Y (loop counter)
    LDA MATH_T16
    BPL Mul8_PosB
    LDA MATH_SIGN
    EOR #1
    STA MATH_SIGN
    LDA MATH_T16
    EOR #$FF
    CLC
    ADC #1
    TAY
    JMP Mul8_Loop
Mul8_PosB:
    TAY                    ; Y = |B|
Mul8_Loop:
    CPY #0
    BEQ Mul8_Sign
    CLC
    LDA MATH_T16_HI
    ADC MATH_T8
    STA MATH_T16_HI        ; product += |A|
    DEY
    JMP Mul8_Loop
Mul8_Sign:
    ; product is in MATH_T16_HI (8-bit magnitude). Move to MATH_T16.
    LDA MATH_T16_HI
    STA MATH_T16
    LDA #0
    STA MATH_T16_HI
    LDA MATH_SIGN
    BEQ Mul8_Done
    ; negate 16-bit result
    LDA MATH_T16
    EOR #$FF
    CLC
    ADC #1
    STA MATH_T16
    LDA MATH_T16_HI
    EOR #$FF
    ADC #0
    STA MATH_T16_HI
Mul8_Done:
    RTS
.endproc

; ------------------------------------------------------------------
; DOT8 — general dot product: sum_i (W[i] * X[i]), 8-bit W, X.
;   In:  MATH_PTR -> W (8-bit signed), MATH_PTR2 -> X (8-bit signed),
;        Y = length (1..255)
;   Out: MATH_T16 (Lo), MATH_T16_HI (Hi) = signed 16-bit sum.
;   Index is kept in DOT_IDX (memory) because MUL8 clobbers registers.
; ------------------------------------------------------------------
.proc DOT8
    STY MATH_CNT           ; length
    LDA #0
    STA DOT_SUM
    STA DOT_SUM_HI
    STA DOT_IDX
Dot8_Loop:
    LDA DOT_IDX
    CMP MATH_CNT
    BEQ Dot8_Done
    TAY                    ; Y = index
    LDA (MATH_PTR),Y      ; A = W[i]
    STA MATH_T8            ; save W[i]
    LDA (MATH_PTR2),Y     ; A = X[i]
    TAY                    ; Y = X[i]
    LDA MATH_T8            ; A = W[i]
    JSR MUL8               ; product in MATH_T16 / MATH_T16_HI
    CLC
    LDA DOT_SUM
    ADC MATH_T16
    STA DOT_SUM
    LDA DOT_SUM_HI
    ADC MATH_T16_HI
    STA DOT_SUM_HI
    INC DOT_IDX
    JMP Dot8_Loop
Dot8_Done:
    LDA DOT_SUM
    STA MATH_T16
    LDA DOT_SUM_HI
    STA MATH_T16_HI
    RTS
.endproc

; ------------------------------------------------------------------
; RELU — ReLU on A. In: A (signed). Out: A = max(0, A).
; ------------------------------------------------------------------
.proc RELU
    BMI Relu_Zero
    RTS
Relu_Zero:
    LDA #0
    RTS
.endproc

; ------------------------------------------------------------------
; QUANTIZE — ternary quantize a latent weight.
;   In:  A = latent (signed 8-bit).
;   Out: A = 1 if latent > 16, -1 if latent < -16, else 0.
;   Dead zone [-16,16] -> 0 (this is what produces sparsity).
; ------------------------------------------------------------------
.proc QUANTIZE
    CMP #17
    BPL Quant_Pos
    ; A < -16 (signed) means A as unsigned >= $F0 AND A != $F0..$FF?
    ; Signed compare: A < -16 iff (A - (-16)) < 0, i.e. A + 16 < 0.
    ; Compute A + 16 and check the sign bit.
    CLC
    ADC #16
    BMI Quant_Neg          ; A + 16 < 0  =>  A < -16
    LDA #0
    RTS
Quant_Pos:
    LDA #1
    RTS
Quant_Neg:
    LDA #$FF               ; -1
    RTS
.endproc

; ------------------------------------------------------------------
; CLAMP_LATENT — clamp A (signed) to [-120, 120].
;   Out: A clamped.
; ------------------------------------------------------------------
.proc CLAMP_LATENT
    CMP #121
    BPL Clamp_Hi           ; A >= 121 (A is positive here, since 121..127)
    ; check A < -120: A + 120 < 0
    CLC
    ADC #120
    BMI Clamp_Lo
    ; restore A (we added 120): subtract 120
    SEC
    SBC #120
    RTS
Clamp_Hi:
    LDA #120
    RTS
Clamp_Lo:
    LDA #$88               ; -120
    RTS
.endproc

; ------------------------------------------------------------------
; EXP_TABLE — exp(-d) * 255 for d = 0..7, d >= 7 -> 0.
;   Used by SOFTMAX_ROW (scores are max-subtracted, so d >= 0).
;   Values: exp(0)=1.0->255, exp(-1)=0.368->94, exp(-2)=0.135->35,
;   exp(-3)=0.0498->13, exp(-4)=0.0183->5, exp(-5)=0.0067->2,
;   exp(-6)=0.0025->1, exp(-7)=0.0009->0.
; ------------------------------------------------------------------
EXP_TABLE:
    .byte 255, 94, 35, 13, 5, 2, 1, 0

; ------------------------------------------------------------------
; SOFTMAX_ROW — 2-element softmax.
;   In:  MATH_PTR -> 2 signed 16-bit scores (Lo,Hi, Lo,Hi).
;   Out: MATH_T8 = w0 (0..255), MATH_T16 = w1 (0..255), w0 + w1 = 255.
;   Method: max-subtract, scale by >>4 (divide by 16) to index the
;   exp table, look up exp, normalize by repeated subtraction.
;   The >>4 is a documented approximation (compresses the score range
;   so the 8-entry table fits).
;   Scratch: $10-$15 (DOT_SUM area, safe here — DOT8 not live).
; ------------------------------------------------------------------
.proc SOFTMAX_ROW
    LDY #0
    LDA (MATH_PTR),Y
    STA $10                ; s0 Lo
    INY
    LDA (MATH_PTR),Y
    STA $11                ; s0 Hi
    INY
    LDA (MATH_PTR),Y
    STA $12                ; s1 Lo
    INY
    LDA (MATH_PTR),Y
    STA $13                ; s1 Hi
    ; max = max(s0, s1): compare Hi first, then Lo
    LDA $11
    CMP $13
    BNE Softmax_HiDiff
    LDA $10
    CMP $12
Softmax_HiDiff:
    BPL Softmax_MaxIsS0
    LDA $12
    STA $14                ; max Lo
    LDA $13
    STA $15                ; max Hi
    JMP Softmax_Diffs
Softmax_MaxIsS0:
    LDA $10
    STA $14
    LDA $11
    STA $15
Softmax_Diffs:
    ; d0 = (max - s0), d1 = (max - s1) — 16-bit subtracts
    SEC
    LDA $14
    SBC $10
    STA $10                ; d0 Lo
    LDA $15
    SBC $11
    STA $11                ; d0 Hi
    SEC
    LDA $14
    SBC $12
    STA $12                ; d1 Lo
    LDA $15
    SBC $13
    STA $13                ; d1 Hi
    ; >>4 on d0 and d1 (arithmetic shift right 4x, 16-bit)
    LDX #4
Softmax_Shift0:
    LDA $11
    ASL                    ; sign bit -> carry
    ROR $11
    ROR $10
    DEX
    BNE Softmax_Shift0
    LDX #4
Softmax_Shift1:
    LDA $13
    ASL
    ROR $13
    ROR $12
    DEX
    BNE Softmax_Shift1
    ; clamp d0, d1 to [0,7]
    LDA $10
    CMP #8
    BCC Softmax_D0Ok
    LDA #7
    STA $10
Softmax_D0Ok:
    LDA $12
    CMP #8
    BCC Softmax_D1Ok
    LDA #7
    STA $12
Softmax_D1Ok:
    ; e0 = EXP_TABLE[d0], e1 = EXP_TABLE[d1]
    LDY $10
    LDA EXP_TABLE,Y
    STA $10                ; e0
    LDY $12
    LDA EXP_TABLE,Y
    STA $12                ; e1
    ; sum = e0 + e1
    CLC
    LDA $10
    ADC $12
    STA $13                ; sum
    BEQ Softmax_Equal
    ; w0 = (e0 * 256) / sum. Compute by repeated subtraction on
    ; the 16-bit value (e0 << 8) = (Hi=e0, Lo=0).
    LDA #0
    STA $14                ; w0
    LDA $10
    STA $15                ; Hi = e0
    LDA #0
    STA $10                ; Lo = 0
Softmax_Div:
    LDA $15
    BNE Softmax_Sub        ; Hi > 0 => value >= 256 > sum, subtract
    LDA $10
    CMP $13
    BCC Softmax_DivDone
Softmax_Sub:
    SEC
    LDA $10
    SBC $13
    STA $10
    LDA $15
    SBC #0
    STA $15
    INC $14
    JMP Softmax_Div
Softmax_DivDone:
    LDA $14
    STA MATH_T8            ; w0
    LDA #255
    SEC
    SBC $14
    STA MATH_T16           ; w1 = 255 - w0
    RTS
Softmax_Equal:
    LDA #127
    STA MATH_T8
    LDA #128
    STA MATH_T16
    RTS
.endproc

; ------------------------------------------------------------------
; TERNARY_CALL — call the user's ternary dot-product kernel.
;   In:  MATH_PTR -> X (input vector, 8-bit), MATH_PTR2 -> W (ternary
;        weights, 8-bit, values -1/0/1), Y = length.
;   Out: A = AccumLo (low byte of the 16-bit sum). AccumHi in $05.
;   Sets the kernel's zero-page pointers and VectorLength, then JSRs
;   TernaryDotProduct (the user-supplied kernel).
; ------------------------------------------------------------------
.proc TERNARY_CALL
    STY VectorLength
    LDA MATH_PTR
    STA InputPtr
    LDA MATH_PTR+1
    STA InputPtr+1
    LDA MATH_PTR2
    STA WeightPtr
    LDA MATH_PTR2+1
    STA WeightPtr+1
    JSR TernaryDotProduct
    LDA AccumLo
    RTS
.endproc

; ------------------------------------------------------------------
; GATHER_COL — gather a column of a 4x4 row-major matrix into a temp.
;   In:  MATH_PTR -> 4x4 matrix (row-major, 16 bytes). Y = column (0..3).
;   Out: GATHER_BUF (4 bytes) = column (rows 0..3).
;   Used for transpose dot products (W^T . v) with the ternary kernel:
;   the kernel needs a contiguous weight vector, so we gather the
;   column into GATHER_BUF, then call the kernel with WeightPtr =
;   GATHER_BUF.
; ------------------------------------------------------------------
GATHER_BUF = $0306          ; 4-byte scratch (in RAM, after LOSS)

.proc GATHER_COL
    TYA
    STA MATH_T8            ; column
    LDX #0
Gather_Loop:
    TXA
    ASL
    ASL                    ; X*4
    CLC
    ADC MATH_T8            ; X*4 + column
    TAY
    LDA (MATH_PTR),Y
    STA GATHER_BUF,X
    INX
    CPX #4
    BNE Gather_Loop
    RTS
.endproc
