; ------------------------------------------------------------------
; 6502 Ternary Dot Product Kernel
; Computes: Accumulator = sum(W_i * X_i) where W_i in {-1, 0, 1}
; ------------------------------------------------------------------
; USER-SUPPLIED CODE — reproduced verbatim. Do not modify.
; This is the core of every ternary projection in the transformer:
; Q, K, V, FFN, and output all call this kernel.
; ------------------------------------------------------------------

; Zero Page Pointers (Set these up before calling)
InputPtr     = $00  ; 16-bit pointer to X vector
WeightPtr    = $02  ; 16-bit pointer to W vector
AccumLo      = $04  ; 16-bit Accumulator (Low Byte)
AccumHi      = $05  ; 16-bit Accumulator (High Byte)
VectorLength = $06  ; Number of elements (up to 255 for this loop)

TernaryDotProduct:
    LDA #0          ; Load 0 into Accumulator
    STA AccumLo     ; Initialize low byte of our sum
    STA AccumHi     ; Initialize high byte of our sum
    LDY #0          ; Y register serves as our loop index

Loop:
    CPY VectorLength  ; Compare Y index to vector length
    BEQ Done          ; If Y == Length, branch to Done

    LDA (WeightPtr),Y ; Load the ternary weight W_i
    BEQ NextElement   ; If W_i == 0, skip math entirely (Massive speedup)

    BMI Subtract      ; If W_i is negative (-1), branch to Subtract

Add:
    ; Handle W_i == 1 (Add X_i to Accumulator)
    LDA (InputPtr),Y  ; Load input activation X_i
    CLC               ; Clear carry flag before addition
    ADC AccumLo       ; Add to Accumulator low byte
    STA AccumLo
    BCC NextElement   ; If no carry over, skip the high byte increment
    INC AccumHi       ; Carry over: increment high byte
    JMP NextElement

Subtract:
    ; Handle W_i == -1 (Subtract X_i from Accumulator)
    LDA AccumLo       ; Load current Accumulator low byte
    SEC               ; Set carry flag before subtraction (6502 rule)
    SBC (InputPtr),Y  ; Subtract input activation X_i
    STA AccumLo
    BCS NextElement   ; If carry is still set (no borrow), skip high byte
    DEC AccumHi       ; Borrow: decrement high byte

NextElement:
    INY               ; Increment Y index
    JMP Loop          ; Loop back for the next parameter

Done:
    RTS               ; Return from Subroutine
