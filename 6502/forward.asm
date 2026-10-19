; ==================================================================
; 6502 TRANSFORMER — FORWARD PASS
; ca65 syntax. Uses the user's ternary kernel (TERNARY_CALL) for
; every ternary projection: Q, K, V, FFN, output logits.
; LayerNorm is OMITTED (documented gap — see README).
; The loss is MSE on the LAST token's logits vs a one-hot target.
; All routines are UNVERIFIED by execution (no 6502 emulator here).
; ==================================================================

; ------------------------------------------------------------------
; FWD_EMBED — X[t] = Emb[token_t] for t = 0,1.
;   INPUT_TOKENS (ROM, 2 bytes) holds the token ids.
;   RAM_EMB (RAM, 4 tokens x 4 = 16 bytes, ternary) holds the table.
; ------------------------------------------------------------------
.proc FWD_EMBED
    LDX #0
FwdEmb_Tok:
    STX FWD_TOK
    ; src = RAM_EMB + token*4
    LDA INPUT_TOKENS,X
    ASL
    ASL
    CLC
    ADC #<RAM_EMB
    STA MATH_PTR
    LDA #>RAM_EMB
    ADC #0
    STA MATH_PTR+1
    ; dst = ACT_X + t*4
    TXA
    ASL
    ASL
    CLC
    ADC #<ACT_X
    STA MATH_PTR2
    LDA #>ACT_X
    ADC #0
    STA MATH_PTR2+1
    LDY #0
FwdEmb_Copy:
    LDA (MATH_PTR),Y
    STA (MATH_PTR2),Y
    INY
    CPY #4
    BNE FwdEmb_Copy
    LDX FWD_TOK
    INX
    CPX #2
    BNE FwdEmb_Tok
    RTS
.endproc

; ------------------------------------------------------------------
; FWD_PROJECT — Q, K, V for both tokens, via the ternary kernel.
;   Q[t][j] = dot(Wq row j, X[t]). Same for K (Wk), V (Wv).
; ------------------------------------------------------------------
.proc FWD_PROJECT
    LDX #0
FwdProj_Tok:
    STX FWD_TOK
    ; X[t] address -> MATH_PTR (input for the kernel)
    TXA
    ASL
    ASL
    CLC
    ADC #<ACT_X
    STA MATH_PTR
    LDA #>ACT_X
    ADC #0
    STA MATH_PTR+1
    LDY #0
FwdProj_Out:
    STY FWD_OUT
    ; --- Q[t][j] ---
    ; WeightPtr = RAM_WQ + j*4
    TYA
    ASL
    ASL
    CLC
    ADC #<RAM_WQ
    STA MATH_PTR2
    LDA #>RAM_WQ
    ADC #0
    STA MATH_PTR2+1
    LDY #4
    JSR TERNARY_CALL      ; A = Q[t][j]
    LDY FWD_OUT
    ; store to ACT_Q + t*4 + j
    LDX FWD_TOK
    TXA
    ASL
    ASL
    CLC
    ADC FWD_OUT
    TAX
    STA ACT_Q,X
    ; --- K[t][j] ---
    LDY FWD_OUT
    TYA
    ASL
    ASL
    CLC
    ADC #<RAM_WK
    STA MATH_PTR2
    LDA #>RAM_WK
    ADC #0
    STA MATH_PTR2+1
    ; MATH_PTR still = X[t] (TERNARY_CALL does not clobber it)
    LDY #4
    JSR TERNARY_CALL
    LDY FWD_OUT
    LDX FWD_TOK
    TXA
    ASL
    ASL
    CLC
    ADC FWD_OUT
    TAX
    STA ACT_K,X
    ; --- V[t][j] ---
    LDY FWD_OUT
    TYA
    ASL
    ASL
    CLC
    ADC #<RAM_WV
    STA MATH_PTR2
    LDA #>RAM_WV
    ADC #0
    STA MATH_PTR2+1
    LDY #4
    JSR TERNARY_CALL
    LDY FWD_OUT
    LDX FWD_TOK
    TXA
    ASL
    ASL
    CLC
    ADC FWD_OUT
    TAX
    STA ACT_V,X
    ; next output j
    LDY FWD_OUT
    INY
    CPY #4
    BNE FwdProj_Out
    LDX FWD_TOK
    INX
    CPX #2
    BNE FwdProj_Tok
    RTS
.endproc

; ------------------------------------------------------------------
; FWD_SCORES — S[i][j] = Q[i] . K[j] (general dot product, 16-bit).
;   Stored to ACT_S + (i*2+j)*2 (Lo,Hi).
; ------------------------------------------------------------------
.proc FWD_SCORES
    LDX #0
FwdS_I:
    STX FWD_I
    LDY #0
FwdS_J:
    STY FWD_J
    ; MATH_PTR = Q[i] = ACT_Q + i*4
    TXA
    ASL
    ASL
    CLC
    ADC #<ACT_Q
    STA MATH_PTR
    LDA #>ACT_Q
    ADC #0
    STA MATH_PTR+1
    ; MATH_PTR2 = K[j] = ACT_K + j*4
    TYA
    ASL
    ASL
    CLC
    ADC #<ACT_K
    STA MATH_PTR2
    LDA #>ACT_K
    ADC #0
    STA MATH_PTR2+1
    LDY #4
    JSR DOT8              ; 16-bit score in MATH_T16/HI
    ; store to ACT_S + (i*2+j)*2
    LDA FWD_I
    ASL
    CLC
    ADC FWD_J
    ASL                   ; (i*2+j)*2
    TAX
    LDA MATH_T16
    STA ACT_S,X
    INX
    LDA MATH_T16_HI
    STA ACT_S,X
    LDY FWD_J
    INY
    CPY #2
    BNE FwdS_J
    LDX FWD_I
    INX
    CPX #2
    BNE FwdS_I
    RTS
.endproc

; ------------------------------------------------------------------
; FWD_SOFTMAX — per-row softmax -> ACT_W (0..255, w0 + w1 = 255).
; ------------------------------------------------------------------
.proc FWD_SOFTMAX
    LDX #0
FwdSm_Row:
    STX FWD_I
    ; MATH_PTR = ACT_S + i*4 (two 16-bit scores)
    TXA
    ASL
    ASL
    CLC
    ADC #<ACT_S
    STA MATH_PTR
    LDA #>ACT_S
    ADC #0
    STA MATH_PTR+1
    JSR SOFTMAX_ROW       ; w0 in MATH_T8, w1 in MATH_T16
    ; store w0, w1 to ACT_W + i*2
    LDX FWD_I
    TXA
    ASL
    TAX
    LDA MATH_T8
    STA ACT_W,X
    INX
    LDA MATH_T16
    STA ACT_W,X
    LDX FWD_I
    INX
    CPX #2
    BNE FwdSm_Row
    RTS
.endproc

; ------------------------------------------------------------------
; FWD_ATTN — O[i][d] = (W[i][0]*V[0][d] + W[i][1]*V[1][d]) >> 8.
;   The >>8 is the high byte of the 16-bit sum (arithmetic shift).
; ------------------------------------------------------------------
.proc FWD_ATTN
    LDX #0
FwdA_I:
    STX FWD_I
    LDY #0
FwdA_D:
    STY FWD_D
    ; term0 = W[i][0] * V[0][d]
    LDX FWD_I
    TXA
    ASL
    TAX
    LDA ACT_W,X           ; W[i][0]
    LDY FWD_D
    LDX ACT_V,Y           ; V[0][d] (ACT_V + 0*4 + d = ACT_V + d)
    ; (ACT_V,Y with Y=d gives V[0][d] since token 0 starts at ACT_V)
    TAY                   ; Y = V[0][d]
    ; A = W[i][0], Y = V[0][d]
    JSR MUL8              ; product0 in MATH_T16/HI
    LDA MATH_T16
    STA $10
    LDA MATH_T16_HI
    STA $11               ; save product0
    ; term1 = W[i][1] * V[1][d]
    LDX FWD_I
    TXA
    ASL
    TAX
    INX                   ; ACT_W + i*2 + 1
    LDA ACT_W,X           ; W[i][1]
    STA $12               ; save W[i][1] (LDA below would clobber A)
    LDY FWD_D
    LDX ACT_V+4,Y         ; V[1][d] (token 1 starts at ACT_V+4)
    TAY                   ; Y = V[1][d]
    LDA $12               ; A = W[i][1]
    JSR MUL8              ; product1 in MATH_T16/HI
    ; sum = product0 + product1
    CLC
    LDA $10
    ADC MATH_T16
    STA $10
    LDA $11
    ADC MATH_T16_HI
    STA $11               ; sum (16-bit) in $10/$11
    ; O[i][d] = sum >> 8 = high byte ($11)
    LDA $11
    ; store to ACT_O + i*4 + d
    LDX FWD_I
    TXA
    ASL
    ASL
    CLC
    ADC FWD_D
    TAX
    STA ACT_O,X
    LDY FWD_D
    INY
    CPY #4
    BNE FwdA_D
    LDX FWD_I
    INX
    CPX #2
    BNE FwdA_I
    RTS
.endproc

; ------------------------------------------------------------------
; FWD_FFN — H1 = ReLU(W1 . O), H = O + W2 . H1 (residual).
;   Per token i, per feature d.
; ------------------------------------------------------------------
.proc FWD_FFN
    LDX #0
FwdF_I:
    STX FWD_I
    ; O[i] address -> MATH_PTR (input for the kernel)
    TXA
    ASL
    ASL
    CLC
    ADC #<ACT_O
    STA MATH_PTR
    LDA #>ACT_O
    ADC #0
    STA MATH_PTR+1
    LDY #0
FwdF_D:
    STY FWD_D
    ; H1[i][d] = ReLU(dot(W1 row d, O[i]))
    TYA
    ASL
    ASL
    CLC
    ADC #<RAM_W1
    STA MATH_PTR2
    LDA #>RAM_W1
    ADC #0
    STA MATH_PTR2+1
    LDY #4
    JSR TERNARY_CALL
    JSR RELU
    LDY FWD_D
    LDX FWD_I
    TXA
    ASL
    ASL
    CLC
    ADC FWD_D
    TAX
    STA ACT_H1,X
    LDY FWD_D
    INY
    CPY #4
    BNE FwdF_D
    ; H[i][d] = O[i][d] + dot(W2 row d, H1[i]) for each d
    ; H1[i] address -> MATH_PTR (input)
    LDX FWD_I
    TXA
    ASL
    ASL
    CLC
    ADC #<ACT_H1
    STA MATH_PTR
    LDA #>ACT_H1
    ADC #0
    STA MATH_PTR+1
    LDY #0
FwdF_D2:
    STY FWD_D
    TYA
    ASL
    ASL
    CLC
    ADC #<RAM_W2
    STA MATH_PTR2
    LDA #>RAM_W2
    ADC #0
    STA MATH_PTR2+1
    LDY #4
    JSR TERNARY_CALL      ; A = dot(W2 row d, H1[i])
    ; H[i][d] = O[i][d] + A
    STA $10                ; save dot
    LDY FWD_D
    LDX FWD_I
    TXA
    ASL
    ASL
    CLC
    ADC FWD_D
    TAX
    LDA ACT_O,X
    CLC
    ADC $10
    STA ACT_H,X
    LDY FWD_D
    INY
    CPY #4
    BNE FwdF_D2
    LDX FWD_I
    INX
    CPX #2
    BNE FwdF_I
    RTS
.endproc

; ------------------------------------------------------------------
; FWD_LOGITS — Logits[i][v] = dot(Wo row v, H[i]).
; ------------------------------------------------------------------
.proc FWD_LOGITS
    LDX #0
FwdL_I:
    STX FWD_I
    ; H[i] address -> MATH_PTR
    TXA
    ASL
    ASL
    CLC
    ADC #<ACT_H
    STA MATH_PTR
    LDA #>ACT_H
    ADC #0
    STA MATH_PTR+1
    LDY #0
FwdL_V:
    STY FWD_OUT
    TYA
    ASL
    ASL
    CLC
    ADC #<RAM_WO
    STA MATH_PTR2
    LDA #>RAM_WO
    ADC #0
    STA MATH_PTR2+1
    LDY #4
    JSR TERNARY_CALL      ; A = Logits[i][v]
    LDY FWD_OUT
    LDX FWD_I
    TXA
    ASL
    ASL
    CLC
    ADC FWD_OUT
    TAX
    STA ACT_LOGITS,X
    LDY FWD_OUT
    INY
    CPY #4
    BNE FwdL_V
    LDX FWD_I
    INX
    CPX #2
    BNE FwdL_I
    RTS
.endproc

; ------------------------------------------------------------------
; FWD_LOSS — MSE on the LAST token (i=1):
;   Loss = sum_v (Logits[1][v] - Target[v])^2.
;   TARGET_VEC (ROM, 4 bytes) is the one-hot target.
;   Loss is 16-bit in LOSS/LOSS+1. May saturate for large logits
;   (documented); in practice logits stay small.
; ------------------------------------------------------------------
.proc FWD_LOSS
    LDA #0
    STA LOSS
    STA LOSS+1
    LDX #0
FwdLoss_V:
    STX FWD_D
    LDA ACT_LOGITS+4,X    ; Logits[1][v]
    SEC
    SBC TARGET_VEC,X
    ; A = diff; square it via MUL8 (A=diff, Y=diff)
    STA MATH_T8
    TAY
    LDA MATH_T8
    JSR MUL8              ; square in MATH_T16/HI
    CLC
    LDA LOSS
    ADC MATH_T16
    STA LOSS
    LDA LOSS+1
    ADC MATH_T16_HI
    STA LOSS+1
    LDX FWD_D
    INX
    CPX #4
    BNE FwdLoss_V
    RTS
.endproc

; ------------------------------------------------------------------
; FORWARD — run the whole forward pass (calls the routines above).
; ------------------------------------------------------------------
.proc FORWARD
    JSR FWD_EMBED
    JSR FWD_PROJECT
    JSR FWD_SCORES
    JSR FWD_SOFTMAX
    JSR FWD_ATTN
    JSR FWD_FFN
    JSR FWD_LOGITS
    JSR FWD_LOSS
    RTS
.endproc
