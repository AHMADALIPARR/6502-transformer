; ==================================================================
; AGC TRANSFORMER -- FORWARD PASS
; embedding -> Q/K/V (ternary dot via kernel) -> scores -> softmax
; -> attention output -> residual + FFN (ternary dot + ReLU)
; -> logits -> MSE loss. LayerNorm is OMITTED (documented gap).
;
; STATUS: WRITTEN, NOT EXECUTED. No AGC emulator here. Do NOT
; claim this runs. Memory map lives in main.agc.
;
; DOT-LOOP STATE MACHINE (all ternary dot products in one loop):
;   DOT_PHASE: 0=Q, 1=K, 2=V, 3=FFN1, 4=FFN2, 5=OUT.
;   DOT_SETUP picks the phase bases (input / weight / dest) from
;   DOT_PHASE, then DOT_CALL stages a kernel call. The kernel
;   returns via RESTORE_EXECUTIVE -> FWD_DOT_CONT, which stores
;   the result, advances pointers/counters, and either calls the
;   kernel again or advances to the next phase (inserting ReLU
;   after FFN1 and the residual after FFN2).
;   RETURN_SLOT 0 = FWD_DOT_CONT (see RET_TABLE in main.agc).
; ==================================================================

; Local constants (data words, forward-only).
SIXTYFOUR       DEC     64

; ------------------------------------------------------------------
; FWD_START -- entry. Embedding lookup, then dot-loop phase 0 (Q).
; ------------------------------------------------------------------
FWD_START:
    ; Embedding: X[t] = EMB[token_t], t = 0..1. Copy 4 words each.
    ; Token 0.
    INDEX   TOKENS
    CA      0               ; token 0
    TS      TMP0            ; TMP0 = token 0
    ; EMB base + token*4. token*4: shift left twice (add to itself).
    CA      TMP0
    ADS     TMP0            ; TMP0 = token*2
    CA      TMP0
    ADS     TMP0            ; TMP0 = token*4
    CA      ADDR_EMB
    ADS     TMP0            ; TMP0 = EMB + token*4 (source base)
    CA      ADDR_X
    TS      TMP1            ; TMP1 = X[0] dest base
    CA      FOUR
    TS      TMP2            ; TMP2 = 4 (counter)
FWD_EMB0:
    INDEX   TMP0
    CA      0
    INDEX   TMP1
    TS      0
    CA      ONE_CONSTANT
    ADS     TMP0
    CA      ONE_CONSTANT
    ADS     TMP1
    CCS     TMP2
    TC      FWD_EMB0
    TC      FWD_EMB1
    TC      FWD_EMB1
FWD_EMB1:
    ; Token 1. TOKENS+1.
    CA      ADDR_TOKENS
    TS      TMP3
    CA      ONE_CONSTANT
    ADS     TMP3
    INDEX   TMP3
    CA      0               ; token 1
    TS      TMP0
    CA      TMP0
    ADS     TMP0
    CA      TMP0
    ADS     TMP0            ; TMP0 = token*4
    CA      ADDR_EMB
    ADS     TMP0            ; TMP0 = EMB + token*4
    CA      ADDR_X
    TS      TMP1
    CA      FOUR
    ADS     TMP1            ; TMP1 = X[1] dest base
    CA      FOUR
    TS      TMP2
FWD_EMB1_LOOP:
    INDEX   TMP0
    CA      0
    INDEX   TMP1
    TS      0
    CA      ONE_CONSTANT
    ADS     TMP0
    CA      ONE_CONSTANT
    ADS     TMP1
    CCS     TMP2
    TC      FWD_EMB1_LOOP
    TC      FWD_DOT_INIT
    TC      FWD_DOT_INIT
FWD_DOT_INIT:
    ; Dot loop: phase 0 (Q). DOT_ADD = 0 (store).
    CA      ZERO_VAL
    TS      DOT_PHASE
    TS      DOT_ADD
    TC      DOT_SETUP

; ------------------------------------------------------------------
; DOT_SETUP -- pick phase bases from DOT_PHASE, then DOT_CALL.
; ------------------------------------------------------------------
DOT_SETUP:
    INDEX   DOT_PHASE
    TC      DOT_SETUP_TBL
DOT_SETUP_TBL:
    TC      SETUP_Q
    TC      SETUP_K
    TC      SETUP_V
    TC      SETUP_F1
    TC      SETUP_F2
    TC      SETUP_OUT
SETUP_Q:
    CA      ADDR_X
    TS      DOT_X_BASE
    TS      DOT_CUR_X
    CA      ADDR_WQ
    TS      DOT_W_BASE
    TS      DOT_CUR_W
    CA      ADDR_Q
    TS      DOT_D_BASE
    TS      DOT_CUR_D
    TC      DOT_SETUP_DONE
SETUP_K:
    CA      ADDR_X
    TS      DOT_X_BASE
    TS      DOT_CUR_X
    CA      ADDR_WK
    TS      DOT_W_BASE
    TS      DOT_CUR_W
    CA      ADDR_K
    TS      DOT_D_BASE
    TS      DOT_CUR_D
    TC      DOT_SETUP_DONE
SETUP_V:
    CA      ADDR_X
    TS      DOT_X_BASE
    TS      DOT_CUR_X
    CA      ADDR_WV
    TS      DOT_W_BASE
    TS      DOT_CUR_W
    CA      ADDR_V
    TS      DOT_D_BASE
    TS      DOT_CUR_D
    TC      DOT_SETUP_DONE
SETUP_F1:
    CA      ADDR_O
    TS      DOT_X_BASE
    TS      DOT_CUR_X
    CA      ADDR_W1
    TS      DOT_W_BASE
    TS      DOT_CUR_W
    CA      ADDR_H
    TS      DOT_D_BASE
    TS      DOT_CUR_D
    TC      DOT_SETUP_DONE
SETUP_F2:
    CA      ADDR_H
    TS      DOT_X_BASE
    TS      DOT_CUR_X
    CA      ADDR_W2
    TS      DOT_W_BASE
    TS      DOT_CUR_W
    CA      ADDR_H2
    TS      DOT_D_BASE
    TS      DOT_CUR_D
    TC      DOT_SETUP_DONE
SETUP_OUT:
    CA      ADDR_H2
    TS      DOT_X_BASE
    TS      DOT_CUR_X
    CA      ADDR_WO
    TS      DOT_W_BASE
    TS      DOT_CUR_W
    CA      ADDR_LOGITS
    TS      DOT_D_BASE
    TS      DOT_CUR_D
    TC      DOT_SETUP_DONE
DOT_SETUP_DONE:
    CA      ZERO_VAL
    TS      DOT_INNER
    TS      DOT_OUTER
    TC      DOT_CALL

; ------------------------------------------------------------------
; DOT_CALL -- stage one kernel call from the current pointers.
; ------------------------------------------------------------------
DOT_CALL:
    CA      DOT_CUR_X
    TS      ARG_X
    CA      DOT_CUR_W
    TS      ARG_W
    CA      FOUR
    TS      ARG_N
    CA      ZERO_VAL        ; RETURN_SLOT 0 = FWD_DOT_CONT
    TS      RETURN_SLOT
    TC      CALL_TERNARY

; ------------------------------------------------------------------
; FWD_DOT_CONT -- kernel returned. Store ACCUM, advance, loop or
; advance phase. (RETURN_SLOT 0.)
; ------------------------------------------------------------------
FWD_DOT_CONT:
    CA      ACCUM_REG_L
    INDEX   DOT_CUR_D
    TS      0               ; dest = dot product
    ; Advance: weight row += 4, dest += 1, inner += 1.
    CA      FOUR
    ADS     DOT_CUR_W
    CA      ONE_CONSTANT
    ADS     DOT_CUR_D
    CA      ONE_CONSTANT
    ADS     DOT_INNER
    ; If inner < 4, next output.
    CA      DOT_INNER
    TS      TMP0
    CS      FOUR
    ADS     TMP0            ; TMP0 = inner - 4
    CA      TMP0
    EXTEND
    BM      DOT_CALL        ; inner < 4: next output
    ; Inner done. Reset inner, outer += 1, input += 4, weight reset.
    CA      ZERO_VAL
    TS      DOT_INNER
    CA      ONE_CONSTANT
    ADS     DOT_OUTER
    CA      FOUR
    ADS     DOT_CUR_X
    CA      DOT_W_BASE
    TS      DOT_CUR_W
    ; If outer < 2, next token.
    CA      DOT_OUTER
    TS      TMP0
    CS      TWO
    ADS     TMP0            ; TMP0 = outer - 2
    CA      TMP0
    EXTEND
    BM      DOT_CALL        ; outer < 2: next token
    ; Phase done. DOT_PHASE += 1. Dispatch on the new phase.
    CA      ONE_CONSTANT
    ADS     DOT_PHASE
    INDEX   DOT_PHASE
    TC      DOT_NEXT_TBL
DOT_NEXT_TBL:
    TC      DOT_SETUP       ; phase 0 (unreached placeholder)
    TC      DOT_SETUP       ; phase 1 -> K
    TC      DOT_SETUP       ; phase 2 -> V
    TC      FWD_SCORES      ; phase 3 -> V done: scores (F1 needs O)
    TC      FWD_RELU        ; phase 4 -> F1 done: ReLU, then FFN2
    TC      FWD_RESID       ; phase 5 -> F2 done: residual, then OUT
    TC      FWD_LOSS        ; phase 6 -> OUT done: loss

; ------------------------------------------------------------------
; FWD_RELU -- ReLU on H (8 values), then DOT_SETUP for phase 4 (FFN2).
; ------------------------------------------------------------------
FWD_RELU:
    CA      ADDR_H
    TS      TMP2            ; TMP2 = H pointer
    CA      POS8
    TS      TMP3            ; TMP3 = 8 (counter)
FWD_RELU_LOOP:
    INDEX   TMP2
    CA      0
    TS      RELU_VAL
    CA      FWD_RELU_CONT
    TS      HELPER_RET
    TC      RELU
FWD_RELU_CONT:
    CA      RELU_VAL
    INDEX   TMP2
    TS      0
    CA      ONE_CONSTANT
    ADS     TMP2
    CCS     TMP3
    TC      FWD_RELU_LOOP
    TC      DOT_SETUP
    TC      DOT_SETUP

; ------------------------------------------------------------------
; FWD_RESID -- H2 += O (8 values), then DOT_SETUP for phase 5 (OUT).
; ------------------------------------------------------------------
FWD_RESID:
    CA      ADDR_H2
    TS      TMP2            ; TMP2 = H2 pointer
    CA      ADDR_O
    TS      TMP3            ; TMP3 = O pointer
    CA      POS8
    TS      TMP0            ; TMP0 = 8 (counter)
FWD_RESID_LOOP:
    INDEX   TMP2
    CA      0
    TS      TMP1            ; TMP1 = H2[i]
    INDEX   TMP3
    CA      0
    ADS     TMP1            ; TMP1 = H2[i] + O[i]
    CA      TMP1
    INDEX   TMP2
    TS      0
    CA      ONE_CONSTANT
    ADS     TMP2
    CA      ONE_CONSTANT
    ADS     TMP3
    CCS     TMP0
    TC      FWD_RESID_LOOP
    TC      DOT_SETUP
    TC      DOT_SETUP

; ------------------------------------------------------------------
; FWD_SCORES -- S[i][j] = sum_d Q[i][d] * K[j][d], 2x2, via MUL8.
; ------------------------------------------------------------------
FWD_SCORES:
    CA      ZERO_VAL
    TS      SCORE_I
    TS      SCORE_J
    TS      SCORE_D
    TS      SCORE_ACC
    CA      ADDR_Q
    TS      SCORE_Q_PTR
    CA      ADDR_K
    TS      SCORE_K_PTR
    CA      ADDR_Q
    TS      TMP1            ; TMP1 = Q row base (Q + i*4)
    CA      ADDR_K
    TS      TMP2            ; TMP2 = K row base (K + j*4)
    CA      ADDR_S
    TS      TMP3            ; TMP3 = S dest pointer
    TC      SCORE_CALL
SCORE_CALL:
    ; MUL_A = Q[SCORE_Q_PTR], MUL_B = K[SCORE_K_PTR].
    INDEX   SCORE_Q_PTR
    CA      0
    TS      MUL_A
    INDEX   SCORE_K_PTR
    CA      0
    TS      MUL_B
    CA      SCORE_CONT
    TS      HELPER_RET
    TC      MUL8
SCORE_CONT:
    ; SCORE_ACC += MUL_R. Advance pointers and d.
    CA      MUL_R
    ADS     SCORE_ACC
    CA      ONE_CONSTANT
    ADS     SCORE_Q_PTR
    CA      ONE_CONSTANT
    ADS     SCORE_K_PTR
    CA      ONE_CONSTANT
    ADS     SCORE_D
    ; If d < 4, next d.
    CA      SCORE_D
    TS      TMP0
    CS      FOUR
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      SCORE_CALL
    ; d done. Store SCORE_ACC to S[i][j].
    CA      SCORE_ACC
    INDEX   TMP3
    TS      0
    CA      ONE_CONSTANT
    ADS     TMP3            ; S pointer += 1
    ; Advance (i, j). j += 1. If j < 2, reset d, Q/K pointers.
    CA      ONE_CONSTANT
    ADS     SCORE_J
    CA      SCORE_J
    TS      TMP0
    CS      TWO
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      SCORE_NEXT_J
    ; j == 2. j = 0, i += 1. If i < 2, reset; else done.
    CA      ZERO_VAL
    TS      SCORE_J
    CA      ONE_CONSTANT
    ADS     SCORE_I
    CA      SCORE_I
    TS      TMP0
    CS      TWO
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      SCORE_NEXT_I
    TC      FWD_SOFTMAX
SCORE_NEXT_J:
    ; Same i, next j. Reset d, ACC. Q ptr = Q row base (TMP1).
    ; K ptr = K row base (TMP2) += 4 (next K row).
    CA      ZERO_VAL
    TS      SCORE_D
    TS      SCORE_ACC
    CA      TMP1
    TS      SCORE_Q_PTR
    CA      FOUR
    ADS     TMP2
    CA      TMP2
    TS      SCORE_K_PTR
    TC      SCORE_CALL
SCORE_NEXT_I:
    ; Next i. Reset d, ACC, j. Q row base += 4. K row base = K.
    CA      ZERO_VAL
    TS      SCORE_D
    TS      SCORE_ACC
    CA      FOUR
    ADS     TMP1
    CA      TMP1
    TS      SCORE_Q_PTR
    CA      ADDR_K
    TS      TMP2
    TS      SCORE_K_PTR
    TC      SCORE_CALL

; ------------------------------------------------------------------
; FWD_SOFTMAX -- softmax per row -> A. Calls SOFT_ROW twice.
; ------------------------------------------------------------------
FWD_SOFTMAX:
    CA      ZERO_VAL
    TS      SOFT_ROW_IDX
    CA      SOFTMAX_ROW1
    TS      SOFT_RET
    TC      SOFT_ROW
SOFTMAX_ROW1:
    CA      ONE_CONSTANT
    TS      SOFT_ROW_IDX
    CA      SOFTMAX_DONE
    TS      SOFT_RET
    TC      SOFT_ROW
SOFTMAX_DONE:
    TC      FWD_ATTN

; SOFT_ROW -- softmax for row SOFT_ROW_IDX. Returns via SOFT_RET.
SOFT_ROW:
    ; SOFT_S_PTR = S + row*2, SOFT_A_PTR = A + row*2.
    CA      ADDR_S
    TS      SOFT_S_PTR
    CA      ADDR_A
    TS      SOFT_A_PTR
    CA      SOFT_ROW_IDX
    EXTEND
    BZ      SOFT_PTR_DONE
    CA      TWO
    ADS     SOFT_S_PTR
    CA      TWO
    ADS     SOFT_A_PTR
SOFT_PTR_DONE:
    ; Load S0 (TMP0), S1 (TMP1).
    INDEX   SOFT_S_PTR
    CA      0
    TS      TMP0
    CA      ONE_CONSTANT
    ADS     SOFT_S_PTR
    INDEX   SOFT_S_PTR
    CA      0
    TS      TMP1
    ; SOFT_MAX = max(S0, S1).
    CA      TMP0
    TS      TMP2
    CS      TMP1
    ADS     TMP2            ; TMP2 = S0 - S1
    CA      TMP2
    EXTEND
    BM      SOFT_MAX_S1
    CA      TMP0
    TS      SOFT_MAX
    TC      SOFT_DIFF0
SOFT_MAX_S1:
    CA      TMP1
    TS      SOFT_MAX
SOFT_DIFF0:
    ; diff0 = S0 - max. Clamp to [-8, 0]. EXP_LOOKUP -> SOFT_E0.
    CA      TMP0
    TS      TMP2
    CS      SOFT_MAX
    ADS     TMP2            ; TMP2 = diff0 (<= 0)
    CA      TMP2
    TS      TMP3
    CA      POS8
    ADS     TMP3            ; TMP3 = diff0 + 8
    CA      TMP3
    EXTEND
    BM      SOFT_DIFF0_CLAMP
    CA      TMP2
    TS      EXP_IN
    TC      SOFT_EXP0
SOFT_DIFF0_CLAMP:
    CA      NEG8
    TS      EXP_IN
SOFT_EXP0:
    CA      SOFT_C1
    TS      HELPER_RET
    TC      EXP_LOOKUP
SOFT_C1:
    CA      EXP_OUT
    TS      SOFT_E0
    ; diff1 = S1 - max. Clamp. EXP_LOOKUP -> SOFT_E1.
    CA      TMP1
    TS      TMP2
    CS      SOFT_MAX
    ADS     TMP2            ; TMP2 = diff1 (<= 0)
    CA      TMP2
    TS      TMP3
    CA      POS8
    ADS     TMP3
    CA      TMP3
    EXTEND
    BM      SOFT_DIFF1_CLAMP
    CA      TMP2
    TS      EXP_IN
    TC      SOFT_EXP1
SOFT_DIFF1_CLAMP:
    CA      NEG8
    TS      EXP_IN
SOFT_EXP1:
    CA      SOFT_C2
    TS      HELPER_RET
    TC      EXP_LOOKUP
SOFT_C2:
    CA      EXP_OUT
    TS      SOFT_E1
    ; SOFT_SUM = E0 + E1.
    CA      SOFT_E0
    TS      SOFT_SUM
    CA      SOFT_E1
    ADS     SOFT_SUM
    ; w0 = (E0 * 64) / SUM -> A[row][0].
    CA      SOFT_E0
    TS      MUL_A
    CA      SIXTYFOUR
    TS      MUL_B
    CA      SOFT_C3
    TS      HELPER_RET
    TC      MUL8
SOFT_C3:
    CA      MUL_R
    TS      DIV_N
    CA      SOFT_SUM
    TS      DIV_D
    CA      SOFT_C4
    TS      HELPER_RET
    TC      DIV
SOFT_C4:
    CA      DIV_Q
    INDEX   SOFT_A_PTR
    TS      0               ; A[row][0] = w0
    CA      ONE_CONSTANT
    ADS     SOFT_A_PTR      ; -> A[row][1]
    ; w1 = (E1 * 64) / SUM -> A[row][1].
    CA      SOFT_E1
    TS      MUL_A
    CA      SIXTYFOUR
    TS      MUL_B
    CA      SOFT_C5
    TS      HELPER_RET
    TC      MUL8
SOFT_C5:
    CA      MUL_R
    TS      DIV_N
    CA      SOFT_SUM
    TS      DIV_D
    CA      SOFT_C6
    TS      HELPER_RET
    TC      DIV
SOFT_C6:
    CA      DIV_Q
    INDEX   SOFT_A_PTR
    TS      0               ; A[row][1] = w1
    INDEX   SOFT_RET
    TC      0

; ------------------------------------------------------------------
; FWD_ATTN -- O[i][d] = (A[i][0]*V[0][d] + A[i][1]*V[1][d]) / 64.
; ------------------------------------------------------------------
FWD_ATTN:
    CA      ZERO_VAL
    TS      ATTN_I
    TS      ATTN_D
    TS      ATTN_ACC
    CA      ADDR_A
    TS      ATTN_A_PTR
    CA      ADDR_V
    TS      ATTN_V_PTR
    CA      ADDR_O
    TS      TMP3            ; TMP3 = O dest pointer
    TC      ATTN_CALL
ATTN_CALL:
    ; MUL_A = A[i][0], MUL_B = V[0][d].
    INDEX   ATTN_A_PTR
    CA      0
    TS      MUL_A
    INDEX   ATTN_V_PTR
    CA      0
    TS      MUL_B
    CA      ATTN_C1
    TS      HELPER_RET
    TC      MUL8
ATTN_C1:
    CA      MUL_R
    TS      ATTN_ACC
    ; MUL_A = A[i][1], MUL_B = V[1][d] (V ptr + 4).
    CA      ATTN_A_PTR
    TS      TMP0
    CA      ONE_CONSTANT
    ADS     TMP0
    INDEX   TMP0
    CA      0
    TS      MUL_A
    CA      ATTN_V_PTR
    TS      TMP0
    CA      FOUR
    ADS     TMP0            ; TMP0 = V[1][d] address
    INDEX   TMP0
    CA      0
    TS      MUL_B
    CA      ATTN_C2
    TS      HELPER_RET
    TC      MUL8
ATTN_C2:
    CA      MUL_R
    ADS     ATTN_ACC
    ; DIV: ATTN_ACC / 64 -> O[i][d].
    CA      ATTN_ACC
    TS      DIV_N
    CA      SIXTYFOUR
    TS      DIV_D
    CA      ATTN_C3
    TS      HELPER_RET
    TC      DIV
ATTN_C3:
    CA      DIV_Q
    INDEX   TMP3
    TS      0               ; O[i][d] = quotient
    CA      ONE_CONSTANT
    ADS     TMP3            ; O pointer += 1
    ; Advance d. If d < 4, next d (V ptr += 1). Else next i.
    CA      ONE_CONSTANT
    ADS     ATTN_D
    CA      ONE_CONSTANT
    ADS     ATTN_V_PTR
    CA      ATTN_D
    TS      TMP0
    CS      FOUR
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      ATTN_CALL
    ; d == 4. d = 0, i += 1. If i < 2, next i. Else done.
    CA      ZERO_VAL
    TS      ATTN_D
    CA      ONE_CONSTANT
    ADS     ATTN_I
    CA      ATTN_I
    TS      TMP0
    CS      TWO
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      ATTN_NEXT_I
    TC      FWD_DOT_FFN
ATTN_NEXT_I:
    ; Next i. A ptr += 2 (next A row). V ptr = V base.
    CA      TWO
    ADS     ATTN_A_PTR
    CA      ADDR_V
    TS      ATTN_V_PTR
    TC      ATTN_CALL

; FWD_DOT_FFN -- re-enter the dot loop for FFN1 (phase 3).
; (Attention output O is ready; FFN1 reads O.)
FWD_DOT_FFN:
    CA      THREE
    TS      DOT_PHASE
    TC      DOT_SETUP

; ------------------------------------------------------------------
; FWD_LOSS -- MSE loss on the last token's logits.
; LOSS = sum_v (LOGITS[1][v] - TARGET_OH[v])^2. Then BWD_START.
; ------------------------------------------------------------------
FWD_LOSS:
    CA      ZERO_VAL
    TS      LOSS_ACC
    TS      LOSS_V
    ; LOGITS[1] base = LOGITS + 4. TARGET_OH base.
    CA      ADDR_LOGITS
    TS      TMP2
    CA      FOUR
    ADS     TMP2            ; TMP2 = LOGITS[1] pointer
    CA      ADDR_TARGET_OH
    TS      TMP3            ; TMP3 = TARGET_OH pointer
    TC      LOSS_CALL
LOSS_CALL:
    ; diff = LOGITS[1][v] - TARGET_OH[v].
    INDEX   TMP2
    CA      0
    TS      TMP0
    INDEX   TMP3
    CA      0
    TS      TMP1
    CA      TMP0
    TS      TMP0
    CS      TMP1
    ADS     TMP0            ; TMP0 = diff
    CA      TMP0
    TS      MUL_A
    TS      MUL_B           ; diff * diff
    CA      LOSS_C1
    TS      HELPER_RET
    TC      MUL8
LOSS_C1:
    CA      MUL_R
    ADS     LOSS_ACC
    CA      ONE_CONSTANT
    ADS     TMP2
    CA      ONE_CONSTANT
    ADS     TMP3
    CA      ONE_CONSTANT
    ADS     LOSS_V
    CA      LOSS_V
    TS      TMP0
    CS      FOUR
    ADS     TMP0
    CA      TMP0
    EXTEND
    BM      LOSS_CALL
    CA      LOSS_ACC
    TS      LOSS
    TC      BWD_START

; Small constant (data word, forward-only).
THREE           DEC     3
