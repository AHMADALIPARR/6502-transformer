; ==================================================================
; 6502 TRANSFORMER — BACKWARD PASS + SGD UPDATE
; ca65 syntax. Gradients are 8-bit signed with documented fixed-point
; approximations (>>8 for attention rescaling, >>4 for score grads)
; to keep values in range. The DIRECTION (sign) of each gradient is
; exact; magnitudes are approximate. This is correct for a tiny demo
; transformer trained by SGD.
;
; Only the LAST token (i=1) contributes to the loss, so backprop
; starts at Logits[1]. Cross-token gradients flow through V (O[1]
; uses V[0] and V[1]) and through K (S[1][*] uses K[0], K[1]).
; Q[0] does not affect the loss, so dQ[0] = 0.
;
; Weight update: latent -= (grad >> 2)  (lr = 1/4, arithmetic shift),
; clamp latent to [-120,120], re-quantize to {-1,0,1} (dead zone
; [-16,16] -> 0), store the ternary back to the RAM weight table.
; All routines are UNVERIFIED by execution (no 6502 emulator here).
; ==================================================================

; --- RAM weight tables (ternary, updated by training) ---
RAM_WQ  = $0500   ; 16 bytes
RAM_WK  = $0510
RAM_WV  = $0520
RAM_W1  = $0530
RAM_W2  = $0540
RAM_WO  = $0550
RAM_EMB = $0560

; --- Latent weights (8-bit, updated by SGD, then quantized) ---
LAT_WQ  = $0400
LAT_WK  = $0410
LAT_WV  = $0420
LAT_W1  = $0430
LAT_W2  = $0440
LAT_WO  = $0450
LAT_EMB = $0460

; ------------------------------------------------------------------
; BWD_DLOGITS — dLogits[1][v] = 2 * (Logits[1][v] - Target[v]).
;   dLogits[0][*] = 0 (token 0 does not contribute to the loss).
; ------------------------------------------------------------------
.proc BWD_DLOGITS
    LDX #0
BwdDl_Zero:
    LDA #0
    STA GRAD_LOGITS,X
    INX
    CPX #4
    BNE BwdDl_Zero
    LDX #0
BwdDl_V:
    LDA ACT_LOGITS+4,X    ; Logits[1][v]
    SEC
    SBC TARGET_VEC,X
    ASL                   ; * 2
    STA GRAD_LOGITS+4,X   ; dLogits[1][v]
    INX
    CPX #4
    BNE BwdDl_V
    RTS
.endproc

; ------------------------------------------------------------------
; BWD_OUTER — generic outer product: G[a][b] = U[a] * V[b].
;   In:  MATH_PTR -> U (4 bytes), MATH_PTR2 -> V (4 bytes),
;        $13 = dest base low, $14 = dest base high (GRAD_* address).
;   Out: 4x4 gradient matrix at dest.
;   Used for dWo, dW2, dW1, dWq, dWk, dWv (with the right U, V).
; ------------------------------------------------------------------
.proc BWD_OUTER
    LDY #0
BwdOuter_A:
    STY FWD_OUT           ; a index
    LDX #0
BwdOuter_B:
    STX FWD_D             ; b index
    ; A = U[a], Y = V[b]
    LDY FWD_OUT
    LDA (MATH_PTR),Y
    STA MATH_T8
    LDY FWD_D
    LDA (MATH_PTR2),Y
    TAY
    LDA MATH_T8
    JSR MUL8
    LDA MATH_T16
    STA $10                ; save product (MUL8 clobbers nothing else)
    ; Y = a*4 + b (dest offset)
    LDY FWD_OUT
    TYA
    ASL
    ASL
    CLC
    ADC FWD_D
    TAY
    LDA $10
    STA ($13),Y
    LDX FWD_D
    INX
    CPX #4
    BNE BwdOuter_B
    LDY FWD_OUT
    INY
    CPY #4
    BNE BwdOuter_A
    RTS
.endproc

; ------------------------------------------------------------------
; BWD_DH — dH[1][d] = sum_v Wo[v][d] * dLogits[1][v] (Wo^T . dLogits).
;   Via GATHER_COL (Wo column d) + TERNARY_CALL (weights = column,
;   input = dLogits[1]). dH[0][*] = 0.
; ------------------------------------------------------------------
.proc BWD_DH
    LDX #0
BwdDh_Zero:
    LDA #0
    STA GRAD_H,X
    INX
    CPX #4
    BNE BwdDh_Zero
    ; MATH_PTR2 = dLogits[1] (input to the kernel)
    LDA #<GRAD_LOGITS+4
    STA MATH_PTR2
    LDA #>GRAD_LOGITS+4
    STA MATH_PTR2+1
    LDY #0
BwdDh_D:
    STY FWD_D
    ; gather Wo column d -> GATHER_BUF
    LDA #<RAM_WO
    STA MATH_PTR
    LDA #>RAM_WO
    STA MATH_PTR+1
    JSR GATHER_COL        ; GATHER_BUF = Wo column d
    ; TERNARY_CALL: input = dLogits[1] (MATH_PTR2), weights = GATHER_BUF
    ; (TERNARY_CALL takes MATH_PTR=input, MATH_PTR2=weights — swap)
    ; Save dLogits ptr, set MATH_PTR = dLogits[1], MATH_PTR2 = GATHER_BUF
    LDA #<GRAD_LOGITS+4
    STA MATH_PTR
    LDA #>GRAD_LOGITS+4
    STA MATH_PTR+1
    LDA #<GATHER_BUF
    STA MATH_PTR2
    LDA #>GATHER_BUF
    STA MATH_PTR2+1
    LDY #4
    JSR TERNARY_CALL      ; A = dH[1][d]
    LDY FWD_D
    STA GRAD_H+4,Y        ; dH[1][d]
    INY
    CPY #4
    BNE BwdDh_D
    RTS
.endproc

; ------------------------------------------------------------------
; BWD_DW2 — dW2[v][d] = dH[1][v] * H1[1][d] (outer product).
;   Uses BWD_OUTER with U = dH[1], V = H1[1], dest = GRAD_W2.
; ------------------------------------------------------------------
.proc BWD_DW2
    LDA #<GRAD_H+4
    STA MATH_PTR
    LDA #>GRAD_H+4
    STA MATH_PTR+1
    LDA #<ACT_H1+4
    STA MATH_PTR2
    LDA #>ACT_H1+4
    STA MATH_PTR2+1
    LDA #<GRAD_W2
    STA $13
    LDA #>GRAD_W2
    STA $14
    JSR BWD_OUTER
    RTS
.endproc

; ------------------------------------------------------------------
; BWD_DH1 — dH1[1][d] = sum_v W2[v][d] * dH[1][v] (W2^T . dH),
;   then ReLU mask: if H1[1][d] <= 0, dH1 = 0.
;   dH1[0][*] = 0.
; ------------------------------------------------------------------
.proc BWD_DH1
    LDX #0
BwdDh1_Zero:
    LDA #0
    STA GRAD_H1,X
    INX
    CPX #4
    BNE BwdDh1_Zero
    LDA #<GRAD_H+4
    STA MATH_PTR2
    LDA #>GRAD_H+4
    STA MATH_PTR2+1
    LDY #0
BwdDh1_D:
    STY FWD_D
    LDA #<RAM_W2
    STA MATH_PTR
    LDA #>RAM_W2
    STA MATH_PTR+1
    JSR GATHER_COL        ; GATHER_BUF = W2 column d
    LDA #<GRAD_H+4
    STA MATH_PTR
    LDA #>GRAD_H+4
    STA MATH_PTR+1
    LDA #<GATHER_BUF
    STA MATH_PTR2
    LDA #>GATHER_BUF
    STA MATH_PTR2+1
    LDY #4
    JSR TERNARY_CALL      ; A = dH1[1][d] (pre-mask)
    ; ReLU mask: if H1[1][d] <= 0, dH1 = 0
    LDY FWD_D
    LDA ACT_H1+4,Y
    BMI BwdDh1_Mask       ; H1 < 0 -> mask
    BEQ BwdDh1_Mask       ; H1 == 0 -> mask (ReLU derivative at 0 is 0)
    ; H1 > 0: keep the gradient (A already holds it)
    JMP BwdDh1_Store
BwdDh1_Mask:
    LDA #0
BwdDh1_Store:
    LDY FWD_D
    STA GRAD_H1+4,Y
    INY
    CPY #4
    BNE BwdDh1_D
    RTS
.endproc

; ------------------------------------------------------------------
; BWD_DW1 — dW1[v][d] = dH1[1][v] * O[1][d] (outer product).
; ------------------------------------------------------------------
.proc BWD_DW1
    LDA #<GRAD_H1+4
    STA MATH_PTR
    LDA #>GRAD_H1+4
    STA MATH_PTR+1
    LDA #<ACT_O+4
    STA MATH_PTR2
    LDA #>ACT_O+4
    STA MATH_PTR2+1
    LDA #<GRAD_W1
    STA $13
    LDA #>GRAD_W1
    STA $14
    JSR BWD_OUTER
    RTS
.endproc

; ------------------------------------------------------------------
; BWD_DO — dO[1][d] = dH[1][d] (residual) + sum_v W1[v][d]*dH1[1][v].
;   The FFN path uses GATHER_COL (W1 column d) + TERNARY_CALL.
;   dO[0][*] = 0.
; ------------------------------------------------------------------
.proc BWD_DO
    LDX #0
BwdDo_Zero:
    LDA #0
    STA GRAD_O,X
    INX
    CPX #4
    BNE BwdDo_Zero
    LDY #0
BwdDo_D:
    STY FWD_D
    ; FFN path: sum_v W1[v][d] * dH1[1][v]
    LDA #<RAM_W1
    STA MATH_PTR
    LDA #>RAM_W1
    STA MATH_PTR+1
    JSR GATHER_COL
    LDA #<GRAD_H1+4
    STA MATH_PTR
    LDA #>GRAD_H1+4
    STA MATH_PTR+1
    LDA #<GATHER_BUF
    STA MATH_PTR2
    LDA #>GATHER_BUF
    STA MATH_PTR2+1
    LDY #4
    JSR TERNARY_CALL      ; A = FFN contribution to dO[1][d]
    ; dO[1][d] = dH[1][d] + A
    CLC
    LDY FWD_D
    ADC GRAD_H+4,Y
    STA GRAD_O+4,Y
    INY
    CPY #4
    BNE BwdDo_D
    RTS
.endproc

; ------------------------------------------------------------------
; BWD_DATNN — attention backward for row i=1 (the loss token).
;   dAttnW[1][j] = sum_d dO[1][d] * V[j][d] >> 8.
;   dV[j][d]     = dO[1][d] * AttnW[1][j] >> 8.
;   dScores via softmax Jacobian (>>8 approximations for /255).
;   dQ[1][d] = sum_j dScores[1][j] * K[j][d] >> 4.
;   dK[j][d] = dScores[1][j] * Q[1][d] >> 4.
; ------------------------------------------------------------------
.proc BWD_DATNN
    ; --- dAttnW[1][j] for j = 0,1 ---
    LDY #0
BwdAttn_J:
    STY FWD_J
    ; sum_d dO[1][d] * V[j][d] (DOT8, then >>8)
    ; MATH_PTR = dO[1] = GRAD_O+4, MATH_PTR2 = V[j] = ACT_V + j*4
    LDA #<GRAD_O+4
    STA MATH_PTR
    LDA #>GRAD_O+4
    STA MATH_PTR+1
    TYA
    ASL
    ASL
    CLC
    ADC #<ACT_V
    STA MATH_PTR2
    LDA #>ACT_V
    ADC #0
    STA MATH_PTR2+1
    LDY #4
    JSR DOT8              ; 16-bit sum in MATH_T16/HI
    ; >>8 = high byte
    LDA MATH_T16_HI
    LDY FWD_J
    STA GRAD_W+2,Y        ; dAttnW[1][j] (row 1 at GRAD_W+2)
    INY
    CPY #2
    BNE BwdAttn_J
    ; --- dV[j][d] = dO[1][d] * AttnW[1][j] >> 8 ---
    LDY #0
BwdDv_J:
    STY FWD_J
    LDX #0
BwdDv_D:
    STX FWD_D
    ; A = dO[1][d], Y = AttnW[1][j]
    LDA GRAD_O+4,X
    STA MATH_T8
    LDY FWD_J
    LDA GRAD_W+2,Y        ; AttnW[1][j]
    TAY
    LDA MATH_T8
    JSR MUL8              ; product in MATH_T16/HI
    LDA MATH_T16_HI       ; >>8
    ; store to GRAD_V + j*4 + d
    LDY FWD_J
    TYA
    ASL
    ASL
    CLC
    ADC FWD_D
    TAX
    STA GRAD_V,X
    LDX FWD_D
    INX
    CPX #4
    BNE BwdDv_D
    LDY FWD_J
    INY
    CPY #2
    BNE BwdDv_J
    ; --- softmax Jacobian for row 1 ---
    ; dot = (w0*g0 + w1*g1) >> 8  (approx /255 as >>8, documented)
    ; w0 = ACT_W+2, g0 = GRAD_W+2, w1 = ACT_W+3, g1 = GRAD_W+3
    LDA ACT_W+2
    LDY GRAD_W+2
    JSR MUL8              ; w0*g0 in MATH_T16/HI
    LDA MATH_T16
    STA $10
    LDA MATH_T16_HI
    STA $11               ; w0*g0
    LDA ACT_W+3
    LDY GRAD_W+3
    JSR MUL8              ; w1*g1
    CLC
    LDA $10
    ADC MATH_T16
    STA $10
    LDA $11
    ADC MATH_T16_HI
    STA $11               ; sum = w0*g0 + w1*g1
    LDA $11
    STA $12               ; dot = sum >> 8
    ; dS0 = w0 * (g0 - dot) >> 8
    LDA GRAD_W+2
    SEC
    SBC $12               ; (g0 - dot)
    TAY                   ; Y = (g0 - dot)
    LDA ACT_W+2           ; A = w0
    JSR MUL8
    LDA MATH_T16_HI       ; >>8
    STA GRAD_S+2          ; dS[1][0]
    ; dS1 = w1 * (g1 - dot) >> 8
    LDA GRAD_W+3
    SEC
    SBC $12               ; (g1 - dot)
    TAY                   ; Y = (g1 - dot)
    LDA ACT_W+3           ; A = w1
    JSR MUL8
    LDA MATH_T16_HI
    STA GRAD_S+3          ; dS[1][1]
    ; --- dQ[1][d] = sum_j dS[1][j] * K[j][d] >> 4 ---
    LDX #0
BwdDq_D:
    STX FWD_D
    ; sum_j dS[1][j] * K[j][d]: j=0 term + j=1 term
    LDA GRAD_S+2          ; dS[1][0]
    LDY ACT_K,X           ; K[0][d] (ACT_K + 0*4 + d)
    STA MATH_T8
    TYA
    TAY
    LDA MATH_T8
    JSR MUL8
    LDA MATH_T16
    STA $10
    LDA MATH_T16_HI
    STA $11               ; term0
    LDA GRAD_S+3          ; dS[1][1]
    LDY ACT_K+4,X         ; K[1][d]
    STA MATH_T8
    TYA
    TAY
    LDA MATH_T8
    JSR MUL8
    CLC
    LDA $10
    ADC MATH_T16
    STA $10
    LDA $11
    ADC MATH_T16_HI
    STA $11               ; sum
    ; >>4 (arithmetic, 16-bit): shift $11:$10 right 4 times
    LDX #4
BwdDq_Shift:
    LDA $11
    ASL
    ROR $11
    ROR $10
    DEX
    BNE BwdDq_Shift
    LDA $10               ; low byte of (sum >> 4)
    LDX FWD_D
    STA GRAD_Q+4,X        ; dQ[1][d]
    INX
    CPX #4
    BNE BwdDq_D
    ; --- dK[j][d] = dS[1][j] * Q[1][d] >> 4 ---
    LDY #0
BwdDk_J:
    STY FWD_J
    LDX #0
BwdDk_D:
    STX FWD_D
    LDA GRAD_S+2,Y        ; dS[1][j] (Y = j)
    STA MATH_T8
    LDA ACT_Q+4,X         ; Q[1][d]
    TAY
    LDA MATH_T8
    JSR MUL8
    ; >>4 on the 16-bit product
    LDA MATH_T16
    STA $10
    LDA MATH_T16_HI
    STA $11
    LDX #4
BwdDk_Shift:
    LDA $11
    ASL
    ROR $11
    ROR $10
    DEX
    BNE BwdDk_Shift
    LDA $10
    ; store to GRAD_K + j*4 + d
    LDY FWD_J
    TYA
    ASL
    ASL
    CLC
    ADC FWD_D
    TAX
    STA GRAD_K,X
    LDX FWD_D
    INX
    CPX #4
    BNE BwdDk_D
    LDY FWD_J
    INY
    CPY #2
    BNE BwdDk_J
    RTS
.endproc

; ------------------------------------------------------------------
; BWD_DWQKV — dWq, dWk, dWv (outer products with X).
;   dWq[v][d] = dQ[1][v] * X[1][d].
;   dWk[v][d] = sum_j dK[j][v] * X[j][d] (j = 0,1).
;   dWv[v][d] = sum_j dV[j][v] * X[j][d] (j = 0,1).
; ------------------------------------------------------------------
.proc BWD_DWQKV
    ; dWq = outer(dQ[1], X[1])
    LDA #<GRAD_Q+4
    STA MATH_PTR
    LDA #>GRAD_Q+4
    STA MATH_PTR+1
    LDA #<ACT_X+4
    STA MATH_PTR2
    LDA #>ACT_X+4
    STA MATH_PTR2+1
    LDA #<GRAD_WQ
    STA $13
    LDA #>GRAD_WQ
    STA $14
    JSR BWD_OUTER
    ; dWk = outer(dK[0], X[0]) + outer(dK[1], X[1])
    ; (j=0 outer into GRAD_WK, j=1 outer into scratch $030A, add)
    LDX #0
BwdDwk_Zero:
    LDA #0
    STA GRAD_WK,X
    INX
    CPX #16
    BNE BwdDwk_Zero
    ; j=0: outer into GRAD_WK
    LDA #<GRAD_K
    STA MATH_PTR
    LDA #>GRAD_K
    STA MATH_PTR+1
    LDA #<ACT_X
    STA MATH_PTR2
    LDA #>ACT_X
    STA MATH_PTR2+1
    LDA #<GRAD_WK
    STA $13
    LDA #>GRAD_WK
    STA $14
    JSR BWD_OUTER
    ; j=1: outer into scratch $030A, then add to GRAD_WK
    LDA #<GRAD_K+4
    STA MATH_PTR
    LDA #>GRAD_K+4
    STA MATH_PTR+1
    LDA #<ACT_X+4
    STA MATH_PTR2
    LDA #>ACT_X+4
    STA MATH_PTR2+1
    LDA #<$030A
    STA $13
    LDA #>$030A
    STA $14
    JSR BWD_OUTER
    LDX #0
BwdDwk_Add:
    CLC
    LDA GRAD_WK,X
    ADC $030A,X
    STA GRAD_WK,X
    INX
    CPX #16
    BNE BwdDwk_Add
    ; dWv = outer(dV[0], X[0]) + outer(dV[1], X[1]) (same pattern)
    LDX #0
BwdDwv_Zero:
    LDA #0
    STA GRAD_WV,X
    INX
    CPX #16
    BNE BwdDwv_Zero
    LDA #<GRAD_V
    STA MATH_PTR
    LDA #>GRAD_V
    STA MATH_PTR+1
    LDA #<ACT_X
    STA MATH_PTR2
    LDA #>ACT_X
    STA MATH_PTR2+1
    LDA #<GRAD_WV
    STA $13
    LDA #>GRAD_WV
    STA $14
    JSR BWD_OUTER
    LDA #<GRAD_V+4
    STA MATH_PTR
    LDA #>GRAD_V+4
    STA MATH_PTR+1
    LDA #<ACT_X+4
    STA MATH_PTR2
    LDA #>ACT_X+4
    STA MATH_PTR2+1
    LDA #<$030A
    STA $13
    LDA #>$030A
    STA $14
    JSR BWD_OUTER
    LDX #0
BwdDwv_Add:
    CLC
    LDA GRAD_WV,X
    ADC $030A,X
    STA GRAD_WV,X
    INX
    CPX #16
    BNE BwdDwv_Add
    RTS
.endproc

; ------------------------------------------------------------------
; BWD_DX — dX from the three projections (transpose dot products).
;   dX[1] = Wq^T.dQ[1] + Wk^T.dK[1] + Wv^T.dV[1].
;   dX[0] = Wk^T.dK[0] + Wv^T.dV[0].   (Q[0] does not affect loss.)
;   Each W^T.v uses GATHER_COL + TERNARY_CALL.
; ------------------------------------------------------------------
.proc BWD_DX
    ; zero dX
    LDX #0
BwdDx_Zero:
    LDA #0
    STA GRAD_X,X
    INX
    CPX #8
    BNE BwdDx_Zero
    ; --- dX[1]: three contributions, accumulate into GRAD_X+4 ---
    LDY #0
BwdDx1_D:
    STY FWD_D
    ; Wq^T.dQ[1] for feature d
    LDA #<RAM_WQ
    STA MATH_PTR
    LDA #>RAM_WQ
    STA MATH_PTR+1
    JSR GATHER_COL
    LDA #<GRAD_Q+4
    STA MATH_PTR
    LDA #>GRAD_Q+4
    STA MATH_PTR+1
    LDA #<GATHER_BUF
    STA MATH_PTR2
    LDA #>GATHER_BUF
    STA MATH_PTR2+1
    LDY #4
    JSR TERNARY_CALL      ; A = Wq^T.dQ[1] for d
    STA $10
    ; Wk^T.dK[1]
    LDA #<RAM_WK
    STA MATH_PTR
    LDA #>RAM_WK
    STA MATH_PTR+1
    LDY FWD_D
    JSR GATHER_COL
    LDA #<GRAD_K+4
    STA MATH_PTR
    LDA #>GRAD_K+4
    STA MATH_PTR+1
    LDA #<GATHER_BUF
    STA MATH_PTR2
    LDA #>GATHER_BUF
    STA MATH_PTR2+1
    LDY #4
    JSR TERNARY_CALL
    CLC
    ADC $10
    STA $10
    ; Wv^T.dV[1]
    LDA #<RAM_WV
    STA MATH_PTR
    LDA #>RAM_WV
    STA MATH_PTR+1
    LDY FWD_D
    JSR GATHER_COL
    LDA #<GRAD_V+4
    STA MATH_PTR
    LDA #>GRAD_V+4
    STA MATH_PTR+1
    LDA #<GATHER_BUF
    STA MATH_PTR2
    LDA #>GATHER_BUF
    STA MATH_PTR2+1
    LDY #4
    JSR TERNARY_CALL
    CLC
    ADC $10
    LDY FWD_D
    STA GRAD_X+4,Y        ; dX[1][d]
    INY
    CPY #4
    BNE BwdDx1_D
    ; --- dX[0]: Wk^T.dK[0] + Wv^T.dV[0] ---
    LDY #0
BwdDx0_D:
    STY FWD_D
    LDA #<RAM_WK
    STA MATH_PTR
    LDA #>RAM_WK
    STA MATH_PTR+1
    JSR GATHER_COL
    LDA #<GRAD_K
    STA MATH_PTR
    LDA #>GRAD_K
    STA MATH_PTR+1
    LDA #<GATHER_BUF
    STA MATH_PTR2
    LDA #>GATHER_BUF
    STA MATH_PTR2+1
    LDY #4
    JSR TERNARY_CALL
    STA $10
    LDA #<RAM_WV
    STA MATH_PTR
    LDA #>RAM_WV
    STA MATH_PTR+1
    LDY FWD_D
    JSR GATHER_COL
    LDA #<GRAD_V
    STA MATH_PTR
    LDA #>GRAD_V
    STA MATH_PTR+1
    LDA #<GATHER_BUF
    STA MATH_PTR2
    LDA #>GATHER_BUF
    STA MATH_PTR2+1
    LDY #4
    JSR TERNARY_CALL
    CLC
    ADC $10
    LDY FWD_D
    STA GRAD_X,Y          ; dX[0][d]
    INY
    CPY #4
    BNE BwdDx0_D
    RTS
.endproc

; ------------------------------------------------------------------
; BWD_DEMB — dEmb[token_t] += dX[t] (accumulate; a token may repeat).
;   Zero GRAD_EMB first, then add dX[t] to the row for token_t.
; ------------------------------------------------------------------
.proc BWD_DEMB
    LDX #0
BwdDemb_Zero:
    LDA #0
    STA GRAD_EMB,X
    INX
    CPX #16
    BNE BwdDemb_Zero
    LDX #0
BwdDemb_Tok:
    STX FWD_TOK
    ; row offset = token * 4
    LDA INPUT_TOKENS,X
    ASL
    ASL
    STA MATH_T8            ; row offset
    LDY #0
BwdDemb_D:
    STY FWD_D
    ; A = dX[t][d] where d = FWD_D, t = FWD_TOK
    LDA FWD_TOK
    ASL
    ASL
    CLC
    ADC FWD_D
    TAX
    LDA GRAD_X,X           ; A = dX[t][d]
    ; dest = GRAD_EMB + row + d
    CLC
    LDY MATH_T8
    TYA
    CLC
    ADC FWD_D
    TAY                    ; Y = row + d
    CLC
    ADC GRAD_EMB,Y
    STA GRAD_EMB,Y
    LDY FWD_D
    INY
    CPY #4
    BNE BwdDemb_D
    LDX FWD_TOK
    INX
    CPX #2
    BNE BwdDemb_Tok
    RTS
.endproc

; ------------------------------------------------------------------
; BWD_UPDATE_ONE — update one 4x4 weight matrix.
;   In:  MATH_PTR -> grad (16 bytes), MATH_PTR2 -> latent (16 bytes),
;        $13/$14 -> ternary RAM table (16 bytes).
;   For each element: latent -= (grad >> 2), clamp, quantize, store.
; ------------------------------------------------------------------
.proc BWD_UPDATE_ONE
    LDX #0
BwdUpd_Elem:
    STX FWD_D
    ; grad = (MATH_PTR)[X]
    LDA (MATH_PTR),X
    ; grad >> 2 (arithmetic shift right 2x)
    STA $10
    ASL                    ; sign -> carry
    ROR $10
    ASL
    ROR $10
    ; latent -= (grad >> 2)
    LDA (MATH_PTR2),X     ; latent
    SEC
    SBC $10
    JSR CLAMP_LATENT
    ; store latent back
    LDX FWD_D
    STA (MATH_PTR2),X
    ; quantize -> ternary, store to RAM table
    JSR QUANTIZE
    LDY FWD_D
    STA ($13),Y
    LDX FWD_D
    INX
    CPX #16
    BNE BwdUpd_Elem
    RTS
.endproc

; ------------------------------------------------------------------
; BWD_UPDATE — update all seven weight tables (Wo, W2, W1, Wq, Wk,
;   Wv, Emb) from their gradients.
; ------------------------------------------------------------------
.proc BWD_UPDATE
    ; Wo
    LDA #<GRAD_WO
    STA MATH_PTR
    LDA #>GRAD_WO
    STA MATH_PTR+1
    LDA #<LAT_WO
    STA MATH_PTR2
    LDA #>LAT_WO
    STA MATH_PTR2+1
    LDA #<RAM_WO
    STA $13
    LDA #>RAM_WO
    STA $14
    JSR BWD_UPDATE_ONE
    ; W2
    LDA #<GRAD_W2
    STA MATH_PTR
    LDA #>GRAD_W2
    STA MATH_PTR+1
    LDA #<LAT_W2
    STA MATH_PTR2
    LDA #>LAT_W2
    STA MATH_PTR2+1
    LDA #<RAM_W2
    STA $13
    LDA #>RAM_W2
    STA $14
    JSR BWD_UPDATE_ONE
    ; W1
    LDA #<GRAD_W1
    STA MATH_PTR
    LDA #>GRAD_W1
    STA MATH_PTR+1
    LDA #<LAT_W1
    STA MATH_PTR2
    LDA #>LAT_W1
    STA MATH_PTR2+1
    LDA #<RAM_W1
    STA $13
    LDA #>RAM_W1
    STA $14
    JSR BWD_UPDATE_ONE
    ; Wq
    LDA #<GRAD_WQ
    STA MATH_PTR
    LDA #>GRAD_WQ
    STA MATH_PTR+1
    LDA #<LAT_WQ
    STA MATH_PTR2
    LDA #>LAT_WQ
    STA MATH_PTR2+1
    LDA #<RAM_WQ
    STA $13
    LDA #>RAM_WQ
    STA $14
    JSR BWD_UPDATE_ONE
    ; Wk
    LDA #<GRAD_WK
    STA MATH_PTR
    LDA #>GRAD_WK
    STA MATH_PTR+1
    LDA #<LAT_WK
    STA MATH_PTR2
    LDA #>LAT_WK
    STA MATH_PTR2+1
    LDA #<RAM_WK
    STA $13
    LDA #>RAM_WK
    STA $14
    JSR BWD_UPDATE_ONE
    ; Wv
    LDA #<GRAD_WV
    STA MATH_PTR
    LDA #>GRAD_WV
    STA MATH_PTR+1
    LDA #<LAT_WV
    STA MATH_PTR2
    LDA #>LAT_WV
    STA MATH_PTR2+1
    LDA #<RAM_WV
    STA $13
    LDA #>RAM_WV
    STA $14
    JSR BWD_UPDATE_ONE
    ; Emb
    LDA #<GRAD_EMB
    STA MATH_PTR
    LDA #>GRAD_EMB
    STA MATH_PTR+1
    LDA #<LAT_EMB
    STA MATH_PTR2
    LDA #>LAT_EMB
    STA MATH_PTR2+1
    LDA #<RAM_EMB
    STA $13
    LDA #>RAM_EMB
    STA $14
    JSR BWD_UPDATE_ONE
    RTS
.endproc

; ------------------------------------------------------------------
; BACKWARD — run the whole backward pass + update.
; ------------------------------------------------------------------
.proc BACKWARD
    JSR BWD_DLOGITS
    ; dWo needs its own outer product (not BWD_OUTER, which uses
    ; MATH_PTR/MATH_PTR2/$13/$14 — BWD_OUTER is generic, so use it)
    LDA #<GRAD_LOGITS+4
    STA MATH_PTR
    LDA #>GRAD_LOGITS+4
    STA MATH_PTR+1
    LDA #<ACT_H+4
    STA MATH_PTR2
    LDA #>ACT_H+4
    STA MATH_PTR2+1
    LDA #<GRAD_WO
    STA $13
    LDA #>GRAD_WO
    STA $14
    JSR BWD_OUTER
    JSR BWD_DH
    JSR BWD_DW2
    JSR BWD_DH1
    JSR BWD_DW1
    JSR BWD_DO
    JSR BWD_DATNN
    JSR BWD_DWQKV
    JSR BWD_DX
    JSR BWD_DEMB
    JSR BWD_UPDATE
    RTS
.endproc
