; ==================================================================
; AGC TRANSFORMER -- BACKWARD PASS + SGD UPDATES
; Gradients through output -> FFN -> attention -> Q/K/V -> X,
; then SGD: latent -= LR * grad, re-quantize to {-1,0,1} (dead zone).
; LayerNorm is OMITTED (documented gap). Embedding table is FROZEN
; (G_X is computed but the embedding lookup is not updated).
;
; STATUS: WRITTEN, NOT EXECUTED. No AGC emulator here. Do NOT
; claim this runs. Memory map lives in main.agc.
;
; The loss is on the LAST token (token 1) only, so gradients flow
; from LOGITS[1]. Token 0's K/V still contribute (token 1 attends
; to token 0), so G_K[0], G_V[0] are nonzero. G_Q[0] is zero.
; The backward dot loop runs both tokens for every phase; token 0
; contributes zero where its input gradient is zero (gradients are
; zeroed at BWD_START), which keeps one uniform loop correct.
;
; BWD dot-loop phases (BWD_PHASE): 0=OUT (G_H2), 1=FFN2 (G_H),
; 2=FFN1 (G_O, ADD), 3=Q (G_X, ADD), 4=K (G_X, ADD), 5=V (G_X, ADD).
; Each phase gathers the transposed weight column into GATHER, then
; dots it with the phase's gradient vector. RETURN_SLOT 1 =
; BWD_DOT_CONT (see RET_TABLE in main.agc).
; ==================================================================

; Local scratch (backward-only). OUTER_* are the outer-product loop.
OUTER_A_PTR     EQU     $00A7
OUTER_B_PTR     EQU     $00A8
OUTER_D_PTR     EQU     $00A9
OUTER_I         EQU     $00AA
OUTER_J         EQU     $00AB
OUTER_ADD       EQU     $00AC
OUTER_RET       EQU     $00AD
BWD_D_PTR       EQU     $00AE           ; backward dot dest pointer
BWD_G_PTR2      EQU     $00AF           ; backward dot grad-vector ptr
ATTN_D2         EQU     $00B0           ; attention-backward d counter
ATTN_J2         EQU     $00B1           ; attention-backward j counter
ATTN_ACC2       EQU     $00B2           ; attention-backward accumulator
ATTN_DOT        EQU     $00B3           ; softmax-Jacobian dot
ATTN_PTR1       EQU     $00B4           ; attention-backward pointer 1
ATTN_PTR2       EQU     $00B5           ; attention-backward pointer 2
ATTN_PTR3       EQU     $00B6           ; attention-backward pointer 3

; ------------------------------------------------------------------
; BWD_START -- zero all gradients, then G_LOGITS.
; Gradient arrays are contiguous G_LOGITS ($0229) .. G_WO end
; ($02D0). Zero them with one loop.
; ------------------------------------------------------------------
BWD_START:
    CA      ADDR_G_LOGITS
    TS      TMP0            ; TMP0 = zero pointer
    CA      ZERO_VAL
    TS      TMP1            ; TMP1 = 0 (value to store)
    ; Count: G_WO end ($02D1) - G_LOGITS ($0229) = 168 words.
    CA      GCOUNT
    TS      TMP2            ; TMP2 = 168
BWD_ZERO_LOOP:
    CA      TMP1
    INDEX   TMP0
    TS      0
    CA      ONE_CONSTANT
    ADS     TMP0
    CCS     TMP2
    TC      BWD_ZERO_LOOP
    TC      BWD_G_LOGITS
    TC      BWD_G_LOGITS
BWD_G_LOGITS:
    ; G_LOGITS[1][v] = 2 * (LOGITS[1][v] - TARGET_OH[v]), v = 0..3.
    ; (G_LOGITS[0] stays 0.)
    CA      ADDR_LOGITS
    TS      TMP0
    CA      FOUR
    ADS     TMP0            ; TMP0 = LOGITS[1] pointer
    CA      ADDR_TARGET_OH
    TS      TMP1            ; TMP1 = TARGET_OH pointer
    CA      ADDR_G_LOGITS
    TS      TMP2
    CA      FOUR
    ADS     TMP2            ; TMP2 = G_LOGITS[1] pointer
    CA      FOUR
    TS      TMP3            ; TMP3 = 4 (counter)
BWD_G_LOGITS_LOOP:
    INDEX   TMP0
    CA      0
    TS      GRAD_TMP        ; GRAD_TMP = LOGITS[1][v]
    INDEX   TMP1
    CA      0
    TS      OUT_TMP         ; OUT_TMP = TARGET_OH[v]
    CA      GRAD_TMP
    TS      GRAD_TMP
    CS      OUT_TMP
    ADS     GRAD_TMP        ; GRAD_TMP = diff
    ; * 2 (add to itself).
    CA      GRAD_TMP
    ADS     GRAD_TMP        ; GRAD_TMP = 2 * diff
    CA      GRAD_TMP
    INDEX   TMP2
    TS      0               ; G_LOGITS[1][v] = 2*diff
    CA      ONE_CONSTANT
    ADS     TMP0
    CA      ONE_CONSTANT
    ADS     TMP1
    CA      ONE_CONSTANT
    ADS     TMP2
    CCS     TMP3
    TC      BWD_G_LOGITS_LOOP
    TC      BWD_G_WO
    TC      BWD_G_WO

; ------------------------------------------------------------------
; BWD_G_WO -- G_WO[v][h] = G_LOGITS[1][v] * H2[1][h] (outer product).
; ------------------------------------------------------------------
BWD_G_WO:
    CA      ADDR_G_LOGITS
    TS      TMP0
    CA      FOUR
    ADS     TMP0            ; TMP0 = G_LOGITS[1]
    TS      OUTER_A_PTR
    CA      ADDR_H2
    TS      TMP0
    CA      FOUR
    ADS     TMP0            ; TMP0 = H2[1]
    TS      OUTER_B_PTR
    CA      ADDR_G_WO
    TS      OUTER_D_PTR
    CA      ZERO_VAL
    TS      OUTER_ADD       ; store (not add)
    CA      BWD_DOT_P0
    TS      OUTER_RET
    TC      OUTER
BWD_DOT_P0:
    ; BWD dot-loop phase 0: G_H2 = WO^T . G_LOGITS.
    CA      ZERO_VAL
    TS      BWD_PHASE
    TC      BWD_DOT_SETUP

; ------------------------------------------------------------------
; BWD_DOT -- one backward dot phase. Gathers the transposed weight
; column into GATHER, dots it with the phase gradient vector.
; BWD_PHASE: 0=OUT, 1=FFN2, 2=FFN1, 3=Q, 4=K, 5=V.
; ------------------------------------------------------------------
BWD_DOT_SETUP:
    INDEX   BWD_PHASE
    TC      BWD_SETUP_TBL
BWD_SETUP_TBL:
    TC      BWD_SET_P0
    TC      BWD_SET_P1
    TC      BWD_SET_P2
    TC      BWD_SET_P3
    TC      BWD_SET_P4
    TC      BWD_SET_P5
BWD_SET_P0:
    ; G_H2 = WO^T . G_LOGITS. Dest G_H2, grad G_LOGITS, W = WO, ADD=0.
    CA      ADDR_G_LOGITS
    TS      BWD_X_BASE
    CA      ADDR_WO
    TS      BWD_W_BASE
    CA      ADDR_G_H2
    TS      BWD_D_BASE
    CA      ZERO_VAL
    TS      BWD_ADD
    TC      BWD_DOT_INIT
BWD_SET_P1:
    ; G_H = W2^T . G_H2. Dest G_H, grad G_H2, W = W2, ADD=0.
    CA      ADDR_G_H2
    TS      BWD_X_BASE
    CA      ADDR_W2
    TS      BWD_W_BASE
    CA      ADDR_G_H
    TS      BWD_D_BASE
    CA      ZERO_VAL
    TS      BWD_ADD
    TC      BWD_DOT_INIT
BWD_SET_P2:
    ; G_O += W1^T . G_H. Dest G_O, grad G_H, W = W1, ADD=1.
    CA      ADDR_G_H
    TS      BWD_X_BASE
    CA      ADDR_W1
    TS      BWD_W_BASE
    CA      ADDR_G_O
    TS      BWD_D_BASE
    CA      ONE_CONSTANT
    TS      BWD_ADD
    TC      BWD_DOT_INIT
BWD_SET_P3:
    ; G_X += WQ^T . G_Q. Dest G_X, grad G_Q, W = WQ, ADD=1.
    CA      ADDR_G_Q
    TS      BWD_X_BASE
    CA      ADDR_WQ
    TS      BWD_W_BASE
    CA      ADDR_G_X
    TS      BWD_D_BASE
    CA      ONE_CONSTANT
    TS      BWD_ADD
    TC      BWD_DOT_INIT
BWD_SET_P4:
    ; G_X += WK^T . G_K. Dest G_X, grad G_K, W = WK, ADD=1.
    CA      ADDR_G_K
    TS      BWD_X_BASE
    CA      ADDR_WK
    TS      BWD_W_BASE
    CA      ADDR_G_X
    TS      BWD_D_BASE
    CA      ONE_CONSTANT
    TS      BWD_ADD
    TC      BWD_DOT_INIT
BWD_SET_P5:
    ; G_X += WV^T . G_V. Dest G_X, grad G_V, W = WV, ADD=1.
    CA      ADDR_G_V
    TS      BWD_X_BASE
    CA      ADDR_WV
    TS      BWD_W_BASE
    CA      ADDR_G_X
    TS      BWD_D_BASE
    CA      ONE_CONSTANT
    TS      BWD_ADD
    TC      BWD_DOT_INIT
BWD_DOT_INIT:
    ; Counters: outer = token (0..1), inner = output dim (0..3).
    CA      ZERO_VAL
    TS      BWD_OUTER
    TS      BWD_INNER
    ; Current grad-vector base = BWD_X_BASE + outer*4.
    CA      BWD_X_BASE
    TS      BWD_G_PTR2
    ; Current dest = BWD_D_BASE + outer*4 (set per token below).
    CA      BWD_D_BASE
    TS      BWD_D_PTR
    TC      BWD_GATHER
BWD_GATHER:
    ; Gather column BWD_INNER of weight matrix BWD_W_BASE into GATHER.
    ; GATHER[v] = W[v*4 + BWD_INNER], v = 0..3.
    CA      BWD_W_BASE
    TS      GATHER_PTR
    CA      BWD_INNER
    ADS     GATHER_PTR      ; GATHER_PTR = W + inner (column offset)
    CA      ADDR_GATHER
    TS      TMP0            ; TMP0 = GATHER dest pointer
    CA      FOUR
    TS      TMP1            ; TMP1 = 4 (counter)
BWD_GATHER_LOOP:
    INDEX   GATHER_PTR
    CA      0
    INDEX   TMP0
    TS      0               ; GATHER[v] = W[v*4 + inner]
    CA      FOUR
    ADS     GATHER_PTR      ; next row (stride 4)
    CA      ONE_CONSTANT
    ADS     TMP0            ; next GATHER slot
    CCS     TMP1
    TC      BWD_GATHER_LOOP
    TC      BWD_DOT_CALL
    TC      BWD_DOT_CALL
BWD_DOT_CALL:
    ; Kernel: dot(GATHER, grad-vector[token]) -> dest[token][inner].
    CA      BWD_G_PTR2
    TS      ARG_X           ; grad vector for this token
    CA      ADDR_GATHER
    TS      ARG_W           ; gathered column
    CA      FOUR
    TS      ARG_N
    CA      ONE_CONSTANT    ; RETURN_SLOT 1 = BWD_DOT_CONT
    TS      RETURN_SLOT
    TC      CALL_TERNARY
BWD_DOT_CONT:
    ; Store or add ACCUM to dest.
    CA      BWD_ADD
    EXTEND
    BZ      BWD_DOT_STORE
    ; ADD mode: dest += ACCUM.
    CA      ACCUM_REG_L
    INDEX   BWD_D_PTR
    ADS     0
    TC      BWD_DOT_ADV
BWD_DOT_STORE:
    CA      ACCUM_REG_L
    INDEX   BWD_D_PTR
    TS      0
BWD_DOT_ADV:
    ; Advance: dest += 1, inner += 1.
    CA      ONE_CONSTANT
    ADS     BWD_D_PTR
    CA      ONE_CONSTANT
    ADS     BWD_INNER
    ; If inner < 4, gather next column (same token).
    CA      BWD_INNER
    TS      TMP0
    CS      FOUR
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      BWD_GATHER
    ; Inner done. Reset inner, outer += 1.
    CA      ZERO_VAL
    TS      BWD_INNER
    CA      ONE_CONSTANT
    ADS     BWD_OUTER
    ; Grad-vector base += 4 (next token). Dest base += 4.
    CA      FOUR
    ADS     BWD_G_PTR2
    CA      FOUR
    ADS     BWD_D_PTR
    ; If outer < 2, gather for next token (inner 0).
    CA      BWD_OUTER
    TS      TMP0
    CS      TWO
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      BWD_GATHER
    ; Phase done. Dispatch on the completed phase.
    INDEX   BWD_PHASE
    TC      BWD_AFTER_TBL
BWD_AFTER_TBL:
    TC      BWD_AFTER_P0
    TC      BWD_AFTER_P1
    TC      BWD_AFTER_P2
    TC      BWD_AFTER_P3
    TC      BWD_AFTER_P4
    TC      BWD_AFTER_P5
BWD_AFTER_P0:
    ; G_H2 done. Residual: G_O[1] = G_H2[1]. Then G_W2. Then phase 1.
    CA      ADDR_G_H2
    TS      TMP0
    CA      FOUR
    ADS     TMP0            ; TMP0 = G_H2[1]
    CA      ADDR_G_O
    TS      TMP1
    CA      FOUR
    ADS     TMP1            ; TMP1 = G_O[1]
    CA      FOUR
    TS      TMP2            ; 4 words
BWD_RESID_LOOP:
    INDEX   TMP0
    CA      0
    INDEX   TMP1
    TS      0
    CA      ONE_CONSTANT
    ADS     TMP0
    CA      ONE_CONSTANT
    ADS     TMP1
    CCS     TMP2
    TC      BWD_RESID_LOOP
    TC      BWD_G_W2
    TC      BWD_G_W2
BWD_G_W2:
    ; G_W2[o][h] = G_H2[1][o] * H[1][h] (outer product).
    CA      ADDR_G_H2
    TS      TMP0
    CA      FOUR
    ADS     TMP0
    TS      OUTER_A_PTR
    CA      ADDR_H
    TS      TMP0
    CA      FOUR
    ADS     TMP0
    TS      OUTER_B_PTR
    CA      ADDR_G_W2
    TS      OUTER_D_PTR
    CA      ZERO_VAL
    TS      OUTER_ADD
    CA      BWD_DOT_P1
    TS      OUTER_RET
    TC      OUTER
BWD_DOT_P1:
    CA      ONE_CONSTANT
    TS      BWD_PHASE
    TC      BWD_DOT_SETUP
BWD_AFTER_P1:
    ; G_H done. ReLU mask, then G_W1, then phase 2.
    TC      BWD_RELU
BWD_RELU:
    ; G_H[1][h] *= (H[1][h] > 0 ? 1 : 0).
    CA      ADDR_H
    TS      TMP0
    CA      FOUR
    ADS     TMP0            ; TMP0 = H[1] pointer
    CA      ADDR_G_H
    TS      TMP1
    CA      FOUR
    ADS     TMP1            ; TMP1 = G_H[1] pointer
    CA      FOUR
    TS      TMP2            ; 4 words
BWD_RELU_LOOP:
    INDEX   TMP0
    CA      0
    EXTEND
    BM      BWD_RELU_ZERO   ; H <= 0 (H is post-ReLU, so H==0 means masked)
    ; H > 0: keep gradient. Advance.
    CA      ONE_CONSTANT
    ADS     TMP0
    CA      ONE_CONSTANT
    ADS     TMP1
    CCS     TMP2
    TC      BWD_RELU_LOOP
    TC      BWD_G_W1
    TC      BWD_G_W1
BWD_RELU_ZERO:
    ; H == 0: zero the gradient.
    CA      ZERO_VAL
    INDEX   TMP1
    TS      0
    CA      ONE_CONSTANT
    ADS     TMP0
    CA      ONE_CONSTANT
    ADS     TMP1
    CCS     TMP2
    TC      BWD_RELU_LOOP
    TC      BWD_G_W1
    TC      BWD_G_W1
BWD_G_W1:
    ; G_W1[h][d] = G_H[1][h] * O[1][d] (outer product).
    CA      ADDR_G_H
    TS      TMP0
    CA      FOUR
    ADS     TMP0
    TS      OUTER_A_PTR
    CA      ADDR_O
    TS      TMP0
    CA      FOUR
    ADS     TMP0
    TS      OUTER_B_PTR
    CA      ADDR_G_W1
    TS      OUTER_D_PTR
    CA      ZERO_VAL
    TS      OUTER_ADD
    CA      BWD_DOT_P2
    TS      OUTER_RET
    TC      OUTER
BWD_DOT_P2:
    CA      TWO
    TS      BWD_PHASE
    TC      BWD_DOT_SETUP
BWD_AFTER_P2:
    ; G_O done. Attention backward, then phase 3 (G_X from Q).
    TC      BWD_ATTN
BWD_AFTER_P3:
    ; G_X from Q done. Phase 4 (G_X from K).
    CA      FOUR
    TS      BWD_PHASE
    TC      BWD_DOT_SETUP
BWD_AFTER_P4:
    ; G_X from K done. Phase 5 (G_X from V).
    CA      FIVE
    TS      BWD_PHASE
    TC      BWD_DOT_SETUP
BWD_AFTER_P5:
    ; G_X from V done. Outer products for WQ/WK/WV, then updates.
    TC      BWD_G_WQKV

; ------------------------------------------------------------------
; BWD_ATTN -- attention backward (token 1 only; token 0's Q unused).
;   G_V[1] += A[1][1] * G_O[1];  G_V[0] += A[1][0] * G_O[1].
;   G_A[1][j] = G_O[1] . V[j].
;   G_S[1][j] = A[1][j] * (G_A[1][j] - sum_k A[1][k]*G_A[1][k]).
;   G_Q[1][d] = G_S[1][0]*K[0][d] + G_S[1][1]*K[1][d].
;   G_K[j][d] = G_S[1][j] * Q[1][d].
; ------------------------------------------------------------------
BWD_ATTN:
    ; --- G_V: for d in 0..3. Two MUL8 per d. ---
    CA      ZERO_VAL
    TS      ATTN_D2
    CA      ADDR_A
    TS      TMP0
    CA      TWO
    ADS     TMP0            ; TMP0 = A[1] base (A + 2; A is 2x2)
    TS      ATTN_PTR1
    CA      ADDR_G_O
    TS      TMP0
    CA      FOUR
    ADS     TMP0            ; TMP0 = G_O[1] base
    TS      ATTN_PTR2
    CA      ADDR_G_V
    TS      ATTN_PTR3         ; ATTN_PTR3 = G_V[0] base
    TC      BWD_ATTN_GV
BWD_ATTN_GV:
    ; MUL_A = A[1][1], MUL_B = G_O[1][d] -> add to G_V[1][d].
    CA      ATTN_PTR1
    TS      TMP0
    CA      ONE_CONSTANT
    ADS     TMP0            ; TMP0 = A[1][1] address
    INDEX   TMP0
    CA      0
    TS      MUL_A
    INDEX   ATTN_PTR2
    CA      0
    TS      MUL_B
    CA      BWD_ATTN_GV_C1
    TS      HELPER_RET
    TC      MUL8
BWD_ATTN_GV_C1:
    ; G_V[1][d] += MUL_R. G_V[1] = G_V[0] + 4.
    CA      ATTN_PTR3
    TS      TMP0
    CA      FOUR
    ADS     TMP0            ; TMP0 = G_V[1][d] address
    CA      MUL_R
    INDEX   TMP0
    ADS     0
    ; MUL_A = A[1][0], MUL_B = G_O[1][d] -> add to G_V[0][d].
    INDEX   ATTN_PTR1
    CA      0
    TS      MUL_A
    INDEX   ATTN_PTR2
    CA      0
    TS      MUL_B
    CA      BWD_ATTN_GV_C2
    TS      HELPER_RET
    TC      MUL8
BWD_ATTN_GV_C2:
    CA      MUL_R
    INDEX   ATTN_PTR3
    ADS     0               ; G_V[0][d] += MUL_R
    ; Advance d. If d < 4, next d.
    CA      ONE_CONSTANT
    ADS     ATTN_PTR2       ; G_O[1] pointer += 1
    CA      ONE_CONSTANT
    ADS     ATTN_PTR3       ; G_V[0] pointer += 1
    CA      ONE_CONSTANT
    ADS     ATTN_D2
    CA      ATTN_D2
    TS      TMP0
    CS      FOUR
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      BWD_ATTN_GV
    ; --- G_A: G_A[1][j] = G_O[1] . V[j], j = 0,1. ---
    TC      BWD_ATTN_GA
BWD_ATTN_GA:
    ; G_A[1][0]: dot(G_O[1], V[0]). G_A[1][1]: dot(G_O[1], V[1]).
    CA      ZERO_VAL
    TS      ATTN_J2
    CA      ADDR_G_A
    TS      TMP0
    CA      TWO
    ADS     TMP0            ; TMP0 = G_A[1] base (G_A + 2; G_A is 2x2)
    TS      ATTN_PTR1
    TC      BWD_ATTN_GA_J
BWD_ATTN_GA_J:
    ; Dot product: sum_d G_O[1][d] * V[j][d]. j = ATTN_J2.
    CA      ZERO_VAL
    TS      ATTN_ACC2
    CA      ZERO_VAL
    TS      ATTN_D2
    CA      ADDR_G_O
    TS      TMP0
    CA      FOUR
    ADS     TMP0            ; TMP0 = G_O[1] pointer
    TS      ATTN_PTR2
    ; V[j] pointer = V + j*4.
    CA      ADDR_V
    TS      ATTN_PTR3
    CA      ATTN_J2
    TS      TMP0
    CA      TMP0
    ADS     TMP0
    CA      TMP0
    ADS     TMP0            ; TMP0 = j*4
    CA      TMP0
    ADS     ATTN_PTR3
    TC      BWD_ATTN_GA_D
BWD_ATTN_GA_D:
    INDEX   ATTN_PTR2
    CA      0
    TS      MUL_A
    INDEX   ATTN_PTR3
    CA      0
    TS      MUL_B
    CA      BWD_ATTN_GA_C
    TS      HELPER_RET
    TC      MUL8
BWD_ATTN_GA_C:
    CA      MUL_R
    ADS     ATTN_ACC2
    CA      ONE_CONSTANT
    ADS     ATTN_PTR2
    CA      ONE_CONSTANT
    ADS     ATTN_PTR3
    CA      ONE_CONSTANT
    ADS     ATTN_D2
    CA      ATTN_D2
    TS      TMP0
    CS      FOUR
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      BWD_ATTN_GA_D
    ; Store ACC2 to G_A[1][j]. Advance j.
    CA      ATTN_ACC2
    INDEX   ATTN_PTR1
    TS      0
    CA      ONE_CONSTANT
    ADS     ATTN_PTR1
    CA      ONE_CONSTANT
    ADS     ATTN_J2
    CA      ATTN_J2
    TS      TMP0
    CS      TWO
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      BWD_ATTN_GA_J
    ; --- Softmax Jacobian dot: dot = A[1][0]*G_A[1][0] + A[1][1]*G_A[1][1]. ---
    TC      BWD_ATTN_DOT
BWD_ATTN_DOT:
    CA      ZERO_VAL
    TS      ATTN_DOT
    CA      ADDR_A
    TS      TMP0
    CA      TWO
    ADS     TMP0            ; TMP0 = A[1] base (A + 2)
    TS      ATTN_PTR1
    CA      ADDR_G_A
    TS      TMP0
    CA      TWO
    ADS     TMP0            ; TMP0 = G_A[1] base (G_A + 2)
    TS      ATTN_PTR2
    CA      ZERO_VAL
    TS      ATTN_D2
    TC      BWD_ATTN_DOT_D
BWD_ATTN_DOT_D:
    INDEX   ATTN_PTR1
    CA      0
    TS      MUL_A
    INDEX   ATTN_PTR2
    CA      0
    TS      MUL_B
    CA      BWD_ATTN_DOT_C
    TS      HELPER_RET
    TC      MUL8
BWD_ATTN_DOT_C:
    CA      MUL_R
    ADS     ATTN_DOT
    CA      ONE_CONSTANT
    ADS     ATTN_PTR1
    CA      ONE_CONSTANT
    ADS     ATTN_PTR2
    CA      ONE_CONSTANT
    ADS     ATTN_D2
    CA      ATTN_D2
    TS      TMP0
    CS      TWO
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      BWD_ATTN_DOT_D
    ; --- G_S: G_S[1][j] = A[1][j] * (G_A[1][j] - dot), j = 0,1. ---
    TC      BWD_ATTN_GS
BWD_ATTN_GS:
    CA      ZERO_VAL
    TS      ATTN_J2
    CA      ADDR_A
    TS      TMP0
    CA      TWO
    ADS     TMP0
    TS      ATTN_PTR1       ; A[1] base (A + 2)
    CA      ADDR_G_A
    TS      TMP0
    CA      TWO
    ADS     TMP0
    TS      ATTN_PTR2       ; G_A[1] base (G_A + 2)
    CA      ADDR_G_S
    TS      TMP0
    CA      TWO
    ADS     TMP0
    TS      ATTN_PTR3       ; G_S[1] base (G_S + 2)
    TC      BWD_ATTN_GS_J
BWD_ATTN_GS_J:
    ; diff = G_A[1][j] - dot.
    INDEX   ATTN_PTR2
    CA      0
    TS      TMP0
    CS      ATTN_DOT
    ADS     TMP0            ; TMP0 = G_A - dot
    ; MUL_A = A[1][j], MUL_B = diff.
    INDEX   ATTN_PTR1
    CA      0
    TS      MUL_A
    CA      TMP0
    TS      MUL_B
    CA      BWD_ATTN_GS_C
    TS      HELPER_RET
    TC      MUL8
BWD_ATTN_GS_C:
    CA      MUL_R
    INDEX   ATTN_PTR3
    TS      0               ; G_S[1][j] = product
    CA      ONE_CONSTANT
    ADS     ATTN_PTR1
    CA      ONE_CONSTANT
    ADS     ATTN_PTR2
    CA      ONE_CONSTANT
    ADS     ATTN_PTR3
    CA      ONE_CONSTANT
    ADS     ATTN_J2
    CA      ATTN_J2
    TS      TMP0
    CS      TWO
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      BWD_ATTN_GS_J
    ; --- G_Q: G_Q[1][d] = G_S[1][0]*K[0][d] + G_S[1][1]*K[1][d]. ---
    TC      BWD_ATTN_GQ
BWD_ATTN_GQ:
    CA      ZERO_VAL
    TS      ATTN_D2
    CA      ADDR_G_S
    TS      TMP0
    CA      TWO
    ADS     TMP0
    TS      ATTN_PTR1       ; G_S[1] base (G_S + 2)
    CA      ADDR_K
    TS      ATTN_PTR2       ; K[0] base
    CA      ADDR_G_Q
    TS      TMP0
    CA      FOUR
    ADS     TMP0
    TS      ATTN_PTR3       ; G_Q[1] base
    TC      BWD_ATTN_GQ_D
BWD_ATTN_GQ_D:
    ; Term 1: G_S[1][0] * K[0][d].
    INDEX   ATTN_PTR1
    CA      0
    TS      MUL_A
    INDEX   ATTN_PTR2
    CA      0
    TS      MUL_B
    CA      BWD_ATTN_GQ_C1
    TS      HELPER_RET
    TC      MUL8
BWD_ATTN_GQ_C1:
    CA      MUL_R
    TS      ATTN_ACC2
    ; Term 2: G_S[1][1] * K[1][d]. K[1][d] = K[0][d] + 4.
    CA      ATTN_PTR1
    TS      TMP0
    CA      ONE_CONSTANT
    ADS     TMP0
    INDEX   TMP0
    CA      0
    TS      MUL_A
    CA      ATTN_PTR2
    TS      TMP0
    CA      FOUR
    ADS     TMP0
    INDEX   TMP0
    CA      0
    TS      MUL_B
    CA      BWD_ATTN_GQ_C2
    TS      HELPER_RET
    TC      MUL8
BWD_ATTN_GQ_C2:
    CA      MUL_R
    ADS     ATTN_ACC2
    CA      ATTN_ACC2
    INDEX   ATTN_PTR3
    TS      0               ; G_Q[1][d] = sum
    CA      ONE_CONSTANT
    ADS     ATTN_PTR2       ; K[0] pointer += 1 (next d)
    CA      ONE_CONSTANT
    ADS     ATTN_PTR3       ; G_Q[1] pointer += 1
    CA      ONE_CONSTANT
    ADS     ATTN_D2
    CA      ATTN_D2
    TS      TMP0
    CS      FOUR
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      BWD_ATTN_GQ_D
    ; --- G_K: G_K[j][d] = G_S[1][j] * Q[1][d], j = 0,1, d = 0..3. ---
    TC      BWD_ATTN_GK
BWD_ATTN_GK:
    CA      ZERO_VAL
    TS      ATTN_J2
    CA      ADDR_G_S
    TS      TMP0
    CA      TWO
    ADS     TMP0
    TS      ATTN_PTR1       ; G_S[1] base (G_S + 2)
    CA      ADDR_Q
    TS      TMP0
    CA      FOUR
    ADS     TMP0
    TS      ATTN_PTR2       ; Q[1] base
    CA      ADDR_G_K
    TS      ATTN_PTR3       ; G_K[0] base
    TC      BWD_ATTN_GK_J
BWD_ATTN_GK_J:
    CA      ZERO_VAL
    TS      ATTN_D2
    TC      BWD_ATTN_GK_D
BWD_ATTN_GK_D:
    INDEX   ATTN_PTR1
    CA      0
    TS      MUL_A           ; G_S[1][j]
    INDEX   ATTN_PTR2
    CA      0
    TS      MUL_B           ; Q[1][d]
    CA      BWD_ATTN_GK_C
    TS      HELPER_RET
    TC      MUL8
BWD_ATTN_GK_C:
    CA      MUL_R
    INDEX   ATTN_PTR3
    TS      0               ; G_K[j][d] = product
    CA      ONE_CONSTANT
    ADS     ATTN_PTR2       ; Q[1] pointer += 1
    CA      ONE_CONSTANT
    ADS     ATTN_PTR3       ; G_K pointer += 1
    CA      ONE_CONSTANT
    ADS     ATTN_D2
    CA      ATTN_D2
    TS      TMP0
    CS      FOUR
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      BWD_ATTN_GK_D
    ; Next j. G_K pointer already advanced by 4 (to next row).
    ; Q[1] pointer: reset to Q[1] base. G_S pointer += 1.
    CA      ADDR_Q
    TS      TMP0
    CA      FOUR
    ADS     TMP0
    TS      ATTN_PTR2
    CA      ONE_CONSTANT
    ADS     ATTN_PTR1
    CA      ONE_CONSTANT
    ADS     ATTN_J2
    CA      ATTN_J2
    TS      TMP0
    CS      TWO
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      BWD_ATTN_GK_J
    ; Attention backward done. Phase 3 (G_X from Q).
    CA      THREE
    TS      BWD_PHASE
    TC      BWD_DOT_SETUP

; ------------------------------------------------------------------
; BWD_G_WQKV -- G_WQ = G_Q[1] (x) X[1]; G_WK = G_K[0](x)X[0] +
; G_K[1](x)X[1]; G_WV = G_V[0](x)X[0] + G_V[1](x)X[1].
; ------------------------------------------------------------------
BWD_G_WQKV:
    ; G_WQ (store).
    CA      ADDR_G_Q
    TS      TMP0
    CA      FOUR
    ADS     TMP0
    TS      OUTER_A_PTR
    CA      ADDR_X
    TS      TMP0
    CA      FOUR
    ADS     TMP0
    TS      OUTER_B_PTR
    CA      ADDR_G_WQ
    TS      OUTER_D_PTR
    CA      ZERO_VAL
    TS      OUTER_ADD
    CA      BWD_G_WK0
    TS      OUTER_RET
    TC      OUTER
BWD_G_WK0:
    ; G_WK = G_K[0] (x) X[0] (store).
    CA      ADDR_G_K
    TS      OUTER_A_PTR
    CA      ADDR_X
    TS      OUTER_B_PTR
    CA      ADDR_G_WK
    TS      OUTER_D_PTR
    CA      ZERO_VAL
    TS      OUTER_ADD
    CA      BWD_G_WK1
    TS      OUTER_RET
    TC      OUTER
BWD_G_WK1:
    ; G_WK += G_K[1] (x) X[1] (add).
    CA      ADDR_G_K
    TS      TMP0
    CA      FOUR
    ADS     TMP0
    TS      OUTER_A_PTR
    CA      ADDR_X
    TS      TMP0
    CA      FOUR
    ADS     TMP0
    TS      OUTER_B_PTR
    CA      ADDR_G_WK
    TS      OUTER_D_PTR
    CA      ONE_CONSTANT
    TS      OUTER_ADD
    CA      BWD_G_WV0
    TS      OUTER_RET
    TC      OUTER
BWD_G_WV0:
    ; G_WV = G_V[0] (x) X[0] (store).
    CA      ADDR_G_V
    TS      OUTER_A_PTR
    CA      ADDR_X
    TS      OUTER_B_PTR
    CA      ADDR_G_WV
    TS      OUTER_D_PTR
    CA      ZERO_VAL
    TS      OUTER_ADD
    CA      BWD_G_WV1
    TS      OUTER_RET
    TC      OUTER
BWD_G_WV1:
    ; G_WV += G_V[1] (x) X[1] (add).
    CA      ADDR_G_V
    TS      TMP0
    CA      FOUR
    ADS     TMP0
    TS      OUTER_A_PTR
    CA      ADDR_X
    TS      TMP0
    CA      FOUR
    ADS     TMP0
    TS      OUTER_B_PTR
    CA      ADDR_G_WV
    TS      OUTER_D_PTR
    CA      ONE_CONSTANT
    TS      OUTER_ADD
    CA      BWD_UPDATE
    TS      OUTER_RET
    TC      OUTER

; ------------------------------------------------------------------
; OUTER -- 4x4 outer product: DST[i][j] = A[i] * B[j] (or += if
; OUTER_ADD == 1). A is OUTER_A_PTR (4 words), B is OUTER_B_PTR,
; DST is OUTER_D_PTR. Returns via OUTER_RET.
; ------------------------------------------------------------------
OUTER:
    CA      ZERO_VAL
    TS      OUTER_I
    TS      OUTER_J
    TC      OUTER_CALL
OUTER_CALL:
    ; MUL_A = A[OUTER_I], MUL_B = B[OUTER_J].
    ; A pointer = OUTER_A_PTR + OUTER_I; B pointer = OUTER_B_PTR + OUTER_J.
    ; (Pointers are advanced in OUTER_CONT; here they point at
    ;  the current (i, j).)
    INDEX   OUTER_A_PTR
    CA      0
    TS      MUL_A
    INDEX   OUTER_B_PTR
    CA      0
    TS      MUL_B
    CA      OUTER_CONT
    TS      HELPER_RET
    TC      MUL8
OUTER_CONT:
    ; Store or add MUL_R to DST.
    CA      OUTER_ADD
    EXTEND
    BZ      OUTER_STORE
    CA      MUL_R
    INDEX   OUTER_D_PTR
    ADS     0
    TC      OUTER_ADV
OUTER_STORE:
    CA      MUL_R
    INDEX   OUTER_D_PTR
    TS      0
OUTER_ADV:
    ; Advance: DST += 1, B pointer += 1, J += 1.
    CA      ONE_CONSTANT
    ADS     OUTER_D_PTR
    CA      ONE_CONSTANT
    ADS     OUTER_B_PTR
    CA      ONE_CONSTANT
    ADS     OUTER_J
    ; If J < 4, next J (same I).
    CA      OUTER_J
    TS      TMP0
    CS      FOUR
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      OUTER_CALL
    ; J == 4. J = 0, I += 1. A pointer += 1 (next A row element).
    ; B pointer: reset to OUTER_B base. DST already advanced by 4.
    CA      ZERO_VAL
    TS      OUTER_J
    CA      ONE_CONSTANT
    ADS     OUTER_I
    ; B pointer -= 4 (back to base). DST pointer: keep (next row).
    CA      OUTER_B_PTR
    TS      TMP0
    CS      FOUR
    ADS     TMP0            ; TMP0 = B_ptr - 4
    CA      TMP0
    TS      OUTER_B_PTR
    ; A pointer += 1 (next A element). But A pointer was advanced?
    ; A pointer is NOT advanced in the inner loop (only B and DST).
    ; So advance A pointer by 1 here (next I).
    CA      ONE_CONSTANT
    ADS     OUTER_A_PTR
    ; If I < 4, next I.
    CA      OUTER_I
    TS      TMP0
    CS      FOUR
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      OUTER_CALL
    ; Done. Return via OUTER_RET.
    INDEX   OUTER_RET
    TC      0

; ------------------------------------------------------------------
; BWD_UPDATE -- SGD: for each of 6 matrices (UPD_PHASE 0..5),
; for each of 16 elements: latent -= LR * grad; weight = QUANTIZE(latent).
; LR = 1, so latent -= grad directly.
; ------------------------------------------------------------------
BWD_UPDATE:
    CA      ZERO_VAL
    TS      UPD_PHASE
    TC      UPD_SETUP
UPD_SETUP:
    INDEX   UPD_PHASE
    TC      UPD_SETUP_TBL
UPD_SETUP_TBL:
    TC      UPD_SET_WQ
    TC      UPD_SET_WK
    TC      UPD_SET_WV
    TC      UPD_SET_W1
    TC      UPD_SET_W2
    TC      UPD_SET_WO
UPD_SET_WQ:
    CA      ADDR_WQ
    TS      UPD_W_PTR
    CA      ADDR_LQ
    TS      UPD_L_PTR
    CA      ADDR_G_WQ
    TS      UPD_G_PTR
    TC      UPD_LOOP_INIT
UPD_SET_WK:
    CA      ADDR_WK
    TS      UPD_W_PTR
    CA      ADDR_LK
    TS      UPD_L_PTR
    CA      ADDR_G_WK
    TS      UPD_G_PTR
    TC      UPD_LOOP_INIT
UPD_SET_WV:
    CA      ADDR_WV
    TS      UPD_W_PTR
    CA      ADDR_LV
    TS      UPD_L_PTR
    CA      ADDR_G_WV
    TS      UPD_G_PTR
    TC      UPD_LOOP_INIT
UPD_SET_W1:
    CA      ADDR_W1
    TS      UPD_W_PTR
    CA      ADDR_L1
    TS      UPD_L_PTR
    CA      ADDR_G_W1
    TS      UPD_G_PTR
    TC      UPD_LOOP_INIT
UPD_SET_W2:
    CA      ADDR_W2
    TS      UPD_W_PTR
    CA      ADDR_L2
    TS      UPD_L_PTR
    CA      ADDR_G_W2
    TS      UPD_G_PTR
    TC      UPD_LOOP_INIT
UPD_SET_WO:
    CA      ADDR_WO
    TS      UPD_W_PTR
    CA      ADDR_LO
    TS      UPD_L_PTR
    CA      ADDR_G_WO
    TS      UPD_G_PTR
    TC      UPD_LOOP_INIT
UPD_LOOP_INIT:
    CA      ZERO_VAL
    TS      UPD_IDX
    TC      UPD_ELEM
UPD_ELEM:
    ; latent -= grad. (LR = 1.)
    INDEX   UPD_L_PTR
    CA      0
    TS      TMP0            ; TMP0 = latent
    INDEX   UPD_G_PTR
    CA      0
    TS      TMP1            ; TMP1 = grad
    CA      TMP0
    TS      TMP0
    CS      TMP1
    ADS     TMP0            ; TMP0 = latent - grad
    CA      TMP0
    INDEX   UPD_L_PTR
    TS      0               ; latent = latent - grad
    ; QUANTIZE(latent) -> weight.
    CA      TMP0
    TS      QUANT_IN
    CA      UPD_CONT
    TS      HELPER_RET
    TC      QUANTIZE
UPD_CONT:
    CA      QUANT_OUT
    INDEX   UPD_W_PTR
    TS      0               ; weight = QUANTIZE(latent)
    ; Advance pointers and index.
    CA      ONE_CONSTANT
    ADS     UPD_W_PTR
    CA      ONE_CONSTANT
    ADS     UPD_L_PTR
    CA      ONE_CONSTANT
    ADS     UPD_G_PTR
    CA      ONE_CONSTANT
    ADS     UPD_IDX
    ; If UPD_IDX < 16, next element.
    CA      UPD_IDX
    TS      TMP0
    CS      SIXTEEN
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      UPD_ELEM
    ; Matrix done. UPD_PHASE += 1. If < 6, next matrix. Else done.
    CA      ONE_CONSTANT
    ADS     UPD_PHASE
    CA      UPD_PHASE
    TS      TMP0
    CS      SIX
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      UPD_SETUP
    ; All matrices updated. Return to the training loop in main.agc.
    TC      TRAIN_AFTER_BWD

; Small constants (data words, backward-only).
FIVE            DEC     5
SIX             DEC     6
GCOUNT          DEC     168             ; gradient words to zero
