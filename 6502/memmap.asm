; ==================================================================
; 6502 TRANSFORMER — MEMORY MAP
; Included FIRST by main.asm so every file can use these labels.
; Model: d_model = 4, seq_len = 2, vocab = 4, 1 layer, 1 head, FFN = 4.
; All activations are 8-bit signed unless noted. Scores are 16-bit.
; ==================================================================

; --- Activations (RAM $0200-$0305) ---
ACT_X       = $0200   ; input vectors, 2 tokens x 4 (8 bytes)
ACT_Q       = $0208   ; Q, 2 x 4 (8)
ACT_K       = $0210   ; K, 2 x 4 (8)
ACT_V       = $0218   ; V, 2 x 4 (8)
ACT_S       = $0220   ; scores, 2 x 2, 16-bit each (8 bytes)
ACT_W       = $0228   ; softmax weights, 2 x 2 (4 bytes, 0..255)
ACT_O       = $022C   ; attention output, 2 x 4 (8)
ACT_H1      = $0234   ; FFN hidden, 2 x 4 (8)
ACT_H       = $023C   ; residual output, 2 x 4 (8)
ACT_LOGITS  = $0244   ; logits, 2 x 4 (8)

; --- Gradients (RAM) ---
GRAD_LOGITS = $024C   ; dL/dlogits, 2 x 4 (8)
GRAD_H      = $0254   ; dL/dH, 2 x 4 (8)
GRAD_H1     = $025C   ; dL/dH1, 2 x 4 (8)
GRAD_O      = $0264   ; dL/dO, 2 x 4 (8)
GRAD_W      = $026C   ; dL/dAttnW, 2 x 2 (4)
GRAD_S      = $0270   ; dL/dScores, 2 x 2 (4)
GRAD_V      = $0274   ; dL/dV, 2 x 4 (8)
GRAD_Q      = $027C   ; dL/dQ, 2 x 4 (8)
GRAD_K      = $0284   ; dL/dK, 2 x 4 (8)
GRAD_X      = $028C   ; dL/dX, 2 x 4 (8)
GRAD_WO     = $0294   ; dL/dWo, 4 x 4 (16)
GRAD_W2     = $02A4   ; dL/dW2, 4 x 4 (16)
GRAD_W1     = $02B4   ; dL/dW1, 4 x 4 (16)
GRAD_WQ     = $02C4   ; dL/dWq, 4 x 4 (16)
GRAD_WK     = $02D4   ; dL/dWk, 4 x 4 (16)
GRAD_WV     = $02E4   ; dL/dWv, 4 x 4 (16)
GRAD_EMB    = $02F4   ; dL/dEmb, 4 x 4 (16)

LOSS        = $0304   ; MSE loss, 16-bit (2 bytes)

; --- Loop counters (zero page, outside the kernel's $00-$06) ---
FWD_TOK     = $13     ; token index
FWD_OUT     = $14     ; output index
FWD_I       = $15     ; row index
FWD_J       = $16     ; column index
FWD_D       = $17     ; feature index
