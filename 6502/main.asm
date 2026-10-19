; ==================================================================
; 6502 TRANSFORMER — MAIN
; A tiny transformer (d_model=4, seq=2, vocab=4, 1 layer, 1 head)
; with a FULL forward + backward pass in 6502 assembly.
; Ternary weights {-1,0,1}. The user's ternary dot-product kernel
; (ternary_dot.asm) is the core of every projection.
;
; Assemble with ca65 (cc65):  ca65 main.asm -o main.o
; Link:                        ld65 -C nrom.cfg main.o -o transformer.nes
; (or assemble all files together with your 6502 assembler of choice;
;  the .include lines below pull the parts in.)
;
; STATUS: Written carefully, UNVERIFIED by execution. There is no
; 6502 emulator on this VM. Do not claim it runs until it has been
; assembled (ca65) and executed (an emulator / hardware) and the
; loss decreases across training iterations.
;
; LayerNorm is OMITTED (documented gap). The residual + ReLU carry
; the signal. See README.md.
; ==================================================================

.segment "CODE"
.org $8000

RESET:
    SEI
    CLD
    LDX #$FF
    TXS                    ; init stack
    JSR INIT
    ; ---- training loop: N iterations of forward + backward ----
    LDX #0
Train_Loop:
    STX $18                ; iteration counter
    JSR FORWARD
    JSR BACKWARD
    LDX $18
    INX
    CPX #8                 ; 8 training iterations
    BNE Train_Loop
Train_Done:
    JMP Train_Done         ; halt (spin). Loss is in LOSS/LOSS+1.

; ------------------------------------------------------------------
; INIT — copy ROM initial weights -> RAM tables, init latents.
;   RAM ternary = ROM ternary (copy 16 bytes per table).
;   Latent = ternary * 64 (so QUANTIZE's dead zone [-16,16] maps
;   back to the same ternary: 64 -> 1, -64 -> -1, 0 -> 0).
; ------------------------------------------------------------------
.proc INIT
    ; (Copy each of the 7 tables: 16 bytes from INIT_* to RAM_*,
    ;  and set the latent.)
    ; EMB
    LDX #0
Init_Emb:
    LDA INIT_EMB,X
    STA RAM_EMB,X
    ; latent = ternary * 64
    CMP #0
    BEQ Init_Emb_Zero
    BMI Init_Emb_Neg
    LDA #64
    JMP Init_Emb_Store
Init_Emb_Neg:
    LDA #$C0               ; -64
    JMP Init_Emb_Store
Init_Emb_Zero:
    LDA #0
Init_Emb_Store:
    STA LAT_EMB,X
    INX
    CPX #16
    BNE Init_Emb
    ; Wq
    LDX #0
Init_Wq:
    LDA INIT_WQ,X
    STA RAM_WQ,X
    CMP #0
    BEQ Init_Wq_Zero
    BMI Init_Wq_Neg
    LDA #64
    JMP Init_Wq_Store
Init_Wq_Neg:
    LDA #$C0
    JMP Init_Wq_Store
Init_Wq_Zero:
    LDA #0
Init_Wq_Store:
    STA LAT_WQ,X
    INX
    CPX #16
    BNE Init_Wq
    ; Wk
    LDX #0
Init_Wk:
    LDA INIT_WK,X
    STA RAM_WK,X
    CMP #0
    BEQ Init_Wk_Zero
    BMI Init_Wk_Neg
    LDA #64
    JMP Init_Wk_Store
Init_Wk_Neg:
    LDA #$C0
    JMP Init_Wk_Store
Init_Wk_Zero:
    LDA #0
Init_Wk_Store:
    STA LAT_WK,X
    INX
    CPX #16
    BNE Init_Wk
    ; Wv
    LDX #0
Init_Wv:
    LDA INIT_WV,X
    STA RAM_WV,X
    CMP #0
    BEQ Init_Wv_Zero
    BMI Init_Wv_Neg
    LDA #64
    JMP Init_Wv_Store
Init_Wv_Neg:
    LDA #$C0
    JMP Init_Wv_Store
Init_Wv_Zero:
    LDA #0
Init_Wv_Store:
    STA LAT_WV,X
    INX
    CPX #16
    BNE Init_Wv
    ; W1
    LDX #0
Init_W1:
    LDA INIT_W1,X
    STA RAM_W1,X
    CMP #0
    BEQ Init_W1_Zero
    BMI Init_W1_Neg
    LDA #64
    JMP Init_W1_Store
Init_W1_Neg:
    LDA #$C0
    JMP Init_W1_Store
Init_W1_Zero:
    LDA #0
Init_W1_Store:
    STA LAT_W1,X
    INX
    CPX #16
    BNE Init_W1
    ; W2
    LDX #0
Init_W2:
    LDA INIT_W2,X
    STA RAM_W2,X
    CMP #0
    BEQ Init_W2_Zero
    BMI Init_W2_Neg
    LDA #64
    JMP Init_W2_Store
Init_W2_Neg:
    LDA #$C0
    JMP Init_W2_Store
Init_W2_Zero:
    LDA #0
Init_W2_Store:
    STA LAT_W2,X
    INX
    CPX #16
    BNE Init_W2
    ; Wo
    LDX #0
Init_Wo:
    LDA INIT_WO,X
    STA RAM_WO,X
    CMP #0
    BEQ Init_Wo_Zero
    BMI Init_Wo_Neg
    LDA #64
    JMP Init_Wo_Store
Init_Wo_Neg:
    LDA #$C0
    JMP Init_Wo_Store
Init_Wo_Zero:
    LDA #0
Init_Wo_Store:
    STA LAT_WO,X
    INX
    CPX #16
    BNE Init_Wo
    RTS
.endproc

; ------------------------------------------------------------------
; DATA — inputs, target, initial weights (ROM).
; Ternary values stored as two's complement: -1 = $FF, 0 = $00, 1 = $01.
; The init patterns are fixed (deterministic, not random) so the
; program is reproducible. They mix -1, 0, 1 so the kernel's
; sparsity (W==0 skip) actually fires.
; ------------------------------------------------------------------
INPUT_TOKENS:
    .byte 0, 1              ; sequence: token 0, then token 1

TARGET_VEC:
    .byte 0, 0, 1, 0        ; one-hot target: token 2 (predict "2")

INIT_EMB:
    .byte  1,  0, $FF,  0
    .byte  0,  1,  0, $FF
    .byte $FF,  0,  1,  0
    .byte  0, $FF,  0,  1

INIT_WQ:
    .byte  1, $FF,  0,  1
    .byte  0,  1, $FF,  0
    .byte $FF,  0,  1, $FF
    .byte  1,  0, $FF,  1

INIT_WK:
    .byte  0,  1, $FF,  1
    .byte  1,  0,  1, $FF
    .byte $FF,  1,  0,  1
    .byte  0, $FF,  1,  0

INIT_WV:
    .byte  1,  1,  0, $FF
    .byte $FF,  0,  1,  1
    .byte  0, $FF,  1,  0
    .byte  1,  0, $FF,  1

INIT_W1:
    .byte  1,  0, $FF,  0
    .byte  0,  1,  0, $FF
    .byte $FF,  0,  1,  0
    .byte  0, $FF,  0,  1

INIT_W2:
    .byte  0,  1,  1, $FF
    .byte  1, $FF,  0,  1
    .byte $FF,  1,  0,  0
    .byte  1,  0, $FF,  1

INIT_WO:
    .byte  1, $FF,  0,  1
    .byte  0,  1, $FF,  0
    .byte $FF,  0,  1, $FF
    .byte  1,  0, $FF,  1

; ------------------------------------------------------------------
; INCLUDES — the parts. Order matters only for readability; ca65
; resolves labels across the whole assembly.
; ------------------------------------------------------------------
.include "memmap.asm"
.include "ternary_dot.asm"
.include "math.asm"
.include "forward.asm"
.include "backward.asm"

; ------------------------------------------------------------------
; VECTORS
; ------------------------------------------------------------------
.segment "VECTORS"
.org $FFFA
    .word 0                ; NMI (unused)
    .word RESET            ; RESET
    .word 0                ; IRQ (unused)
