; ==================================================================
; AGC TRANSFORMER -- MAIN (entry, memory map, data, training loop)
; Tiny transformer, full forward + backward pass, AGC assembly only.
;
; STATUS: WRITTEN, NOT EXECUTED. No AGC emulator is available on this
; machine, so this code has NOT been run on a real AGC or yaAGC.
; Do NOT claim it runs. Read it, assemble it with yaAGC, and test it.
;
; MODEL (same as the 6502 half, so the two are comparable):
;   d_model = 4, seq_len = 2, vocab = 4, 1 layer, 1 attention head,
;   FFN hidden = 4. All weights TERNARY (-1, 0, 1).
;   Forward: embedding -> Q/K/V (ternary dot) -> scores -> softmax
;   -> attention output -> residual + FFN (ternary dot + ReLU)
;   -> logits -> MSE loss. Backward: gradients + SGD with latent
;   weights + dead-zone re-quantization. LayerNorm is OMITTED
;   (documented gap -- it is NOT faked).
;
; AGC CONSTRAINTS (honest):
;   15-bit words, ones' complement arithmetic. No hardware
;   multiply/divide (MUL/DIV are done by repeated add/subtract
;   in math.asm). No stack. Fixed ROM/RAM partitions. Zero dynamic
;   allocation. Deterministic branches. Small values are kept so
;   products stay inside 15 bits; -0 is avoided by construction
;   (data is initialized to +0, never -0).
;
; CONTROL FLOW (no stack, no Q-register returns):
;   - Top-level phases are entered with TC and fall through / TC
;     to the next phase explicitly. No routine relies on Q for return.
;   - The kernel (ternary_squib.asm) returns via TCF
;     RESTORE_EXECUTIVE (its own fixed return). RESTORE_EXECUTIVE
;     dispatches through RET_TABLE using RETURN_SLOT, which the
;     caller sets before each kernel call.
;   - Helpers in math.asm return via HELPER_RET: the caller sets
;     HELPER_RET to a continuation label, then TC HELPER. The
;     helper ends with INDEX HELPER_RET / TC 0.
;   - CALL_TERNARY stages operands into the kernel's fixed buffers
;     ($1000 input, $2000 weights) and sets VECTOR_COUNT, then
;     TCs to APOLLO_TERNARY_REDUCTION. On return (via the
;     dispatcher) ACCUM_REG_L holds the dot product; the caller's
;     continuation reads it.
;   - The kernel's $1000/$2000 are its fixed input buffers per its
;     own map. The wrapper stages each dot-product's operands there.
;     (On a real AGC these are fixed-memory addresses; the repo
;     treats them as the kernel's input buffers, per the kernel.)
;
; FILES (assemble together; labels are shared across files):
;   ternary_squib.asm -- user-supplied kernel (VERBATIM, do not edit)
;   math.asm          -- MUL/DIV/EXP/RELU/CLAMP/QUANTIZE/SIGN/ABS
;   forward.asm       -- forward pass
;   backward.asm      -- backward pass + SGD updates
;   main.agc          -- this file: map, data, wrapper, loop, HALT
; ==================================================================

; ------------------------------------------------------------------
; MEMORY MAP -- scratch (erasable). Kernel owns $0040-$0043.
; ------------------------------------------------------------------
ZERO_VAL        EQU     $0044           ; holds +0
LOOP_COUNTER    EQU     $0045           ; kernel loop index (kernel extern)
ONE_CONSTANT    EQU     $0046           ; holds 1 (kernel extern)
RETURN_SLOT     EQU     $0047           ; kernel return dispatch ID
HELPER_RET      EQU     $0048           ; math-helper return address
SOFT_RET        EQU     $0049           ; softmax-row return address
COPY_SRC        EQU     $004A           ; wrapper copy source pointer
COPY_DST        EQU     $004B           ; wrapper copy dest pointer
COPY_CNT        EQU     $004C           ; wrapper copy counter
PTR_ZBUF        EQU     $004D           ; holds $1000 (kernel input buffer)
PTR_WBUF        EQU     $004E           ; holds $2000 (kernel weight buffer)
BUDGET_VAL      EQU     $004F           ; CYCLE_BUDGET reload value

TMP0            EQU     $0050
TMP1            EQU     $0051
TMP2            EQU     $0052
TMP3            EQU     $0053
DOT_PHASE       EQU     $0054           ; dot-loop phase (0..5 forward)
DOT_CUR_X       EQU     $0055           ; dot-loop current input base
DOT_CUR_W       EQU     $0056           ; dot-loop current weight row base
DOT_CUR_D       EQU     $0057           ; dot-loop current destination
DOT_INNER       EQU     $0058           ; dot-loop inner counter (output dim)
DOT_OUTER       EQU     $0059           ; dot-loop outer counter (token)
DOT_ADD         EQU     $005A           ; 0=store result, 1=add result
DOT_X_BASE      EQU     $005B           ; dot-loop phase input base
DOT_W_BASE      EQU     $005C           ; dot-loop phase weight base
DOT_D_BASE      EQU     $005D           ; dot-loop phase dest base
ARG_X           EQU     $005E           ; wrapper arg: input vector base
ARG_W           EQU     $005F           ; wrapper arg: weight row base
ARG_N           EQU     $0060           ; wrapper arg: length
MUL_A           EQU     $0061
MUL_B           EQU     $0062
MUL_R           EQU     $0063
MUL_A_ABS       EQU     $0064
MUL_B_ABS       EQU     $0065
MUL_SIGN        EQU     $0066
DIV_N           EQU     $0067
DIV_D           EQU     $0068
DIV_Q           EQU     $0069
DIV_R           EQU     $006A
EXP_IN          EQU     $006B
EXP_OUT         EQU     $006C
EXP_IDX         EQU     $006D
RELU_VAL        EQU     $006E
CLAMP_IN        EQU     $006F
CLAMP_OUT       EQU     $0070
QUANT_IN        EQU     $0071
QUANT_OUT       EQU     $0072
SIGN_IN         EQU     $0073
SIGN_OUT        EQU     $0074
ABS_IN          EQU     $0075
ABS_OUT         EQU     $0076
SCORE_I         EQU     $0077
SCORE_J         EQU     $0078
SCORE_D         EQU     $0079
SCORE_ACC       EQU     $007A
SCORE_Q_PTR     EQU     $007B
SCORE_K_PTR     EQU     $007C
SOFT_ROW_IDX    EQU     $007D
SOFT_S_PTR      EQU     $007E
SOFT_A_PTR      EQU     $007F
SOFT_MAX        EQU     $0080
SOFT_E0         EQU     $0081
SOFT_E1         EQU     $0082
SOFT_SUM        EQU     $0083
SOFT_DIFF       EQU     $0084
ATTN_I          EQU     $0085
ATTN_D          EQU     $0086
ATTN_ACC        EQU     $0087
ATTN_A_PTR      EQU     $0088
ATTN_V_PTR      EQU     $0089
LOSS_ACC        EQU     $008A
LOSS_V          EQU     $008B
GATHER_W_BASE   EQU     $008C           ; gather: weight matrix base
GATHER_H        EQU     $008D           ; gather: column index
GATHER_PTR      EQU     $008E           ; gather: running pointer
BWD_PHASE       EQU     $008F           ; backward dot-loop phase
BWD_CUR_X       EQU     $0090           ; backward dot-loop input base
BWD_CUR_W       EQU     $0091           ; backward dot-loop weight base (GATHER)
BWD_CUR_D       EQU     $0092           ; backward dot-loop destination
BWD_INNER       EQU     $0093           ; backward dot-loop inner counter
BWD_OUTER       EQU     $0094           ; backward dot-loop outer counter
BWD_ADD         EQU     $0095           ; 0=store, 1=add (accumulate into G_X)
BWD_X_BASE      EQU     $0096           ; backward dot-loop grad-vector base
BWD_W_BASE      EQU     $0097           ; backward dot-loop weight matrix base
BWD_D_BASE      EQU     $0098           ; backward dot-loop dest base
BWD_G_PTR       EQU     $0099           ; backward: current grad-vector base
ITER            EQU     $009A           ; training iteration counter
LR_VAL          EQU     $009B           ; learning rate (integer step)
DEAD_VAL        EQU     $009C           ; dead-zone half-width (16)
DEADP1          EQU     $009D           ; dead-zone + 1 (17)
POS8            EQU     $009E           ; 8
NEG8            EQU     $009F           ; -8 (ones' complement)
UPD_W_PTR       EQU     $00A0           ; update: weight pointer
UPD_L_PTR       EQU     $00A1           ; update: latent pointer
UPD_G_PTR       EQU     $00A2           ; update: gradient pointer
UPD_IDX         EQU     $00A3           ; update: element counter
UPD_PHASE       EQU     $00A4           ; update: which matrix (0..5)
GRAD_TMP        EQU     $00A5           ; gradient temp
OUT_TMP         EQU     $00A6           ; output temp

; ------------------------------------------------------------------
; MEMORY MAP -- data tables (erasable, fixed).
; ------------------------------------------------------------------
EMB             EQU     $0100           ; embedding, 4 tokens x 4 dims
WQ              EQU     $0110           ; Q projection, 4x4 ternary
WK              EQU     $0120           ; K projection, 4x4
WV              EQU     $0130           ; V projection, 4x4
W1              EQU     $0140           ; FFN layer 1, 4x4
W2              EQU     $0150           ; FFN layer 2, 4x4
WO              EQU     $0160           ; output projection, 4x4
LQ              EQU     $0170           ; latent Q weights, 4x4
LK              EQU     $0180           ; latent K weights, 4x4
LV              EQU     $0190           ; latent V weights, 4x4
L1              EQU     $01A0           ; latent FFN1 weights, 4x4
L2              EQU     $01B0           ; latent FFN2 weights, 4x4
LO              EQU     $01C0           ; latent output weights, 4x4
TOKENS          EQU     $01D0           ; input tokens, 2 words
TARGET_OH       EQU     $01D2           ; target one-hot, 4 words
TARGET_TOK      EQU     $01D6           ; target token index
X               EQU     $01E0           ; embedded inputs, 2 x 4
Q               EQU     $01E8           ; queries, 2 x 4
K               EQU     $01F0           ; keys, 2 x 4
V               EQU     $01F8           ; values, 2 x 4
S               EQU     $0200           ; attention scores, 2 x 2
A               EQU     $0204           ; attention weights, 2 x 2
O               EQU     $0208           ; attention output, 2 x 4
H               EQU     $0210           ; FFN hidden (after ReLU), 2 x 4
H2              EQU     $0218           ; FFN output + residual, 2 x 4
LOGITS          EQU     $0220           ; output logits, 2 x 4
LOSS            EQU     $0228           ; MSE loss (scalar)
G_LOGITS        EQU     $0229           ; grad logits, 2 x 4
G_H2            EQU     $0231           ; grad H2, 2 x 4
G_H             EQU     $0239           ; grad H (FFN hidden), 2 x 4
G_O             EQU     $0241           ; grad attention output, 2 x 4
G_V             EQU     $0249           ; grad values, 2 x 4
G_S             EQU     $0251           ; grad scores, 2 x 2
G_A             EQU     $0255           ; grad attention weights, 2 x 2
G_Q             EQU     $0259           ; grad queries, 2 x 4
G_K             EQU     $0261           ; grad keys, 2 x 4
G_X             EQU     $0269           ; grad embedded inputs, 2 x 4
G_WQ            EQU     $0271           ; grad WQ, 4x4
G_WK            EQU     $0281           ; grad WK, 4x4
G_WV            EQU     $0291           ; grad WV, 4x4
G_W1            EQU     $02A1           ; grad W1, 4x4
G_W2            EQU     $02B1           ; grad W2, 4x4
G_WO            EQU     $02C1           ; grad WO, 4x4
GATHER          EQU     $02D1           ; gathered weight column, 4 words

; ------------------------------------------------------------------
; ADDRESS CELLS -- data words holding base addresses, so the code
; can pass pointers as values (AGC has no immediate addresses).
; Values are the decimal equivalents of the hex bases above.
; ------------------------------------------------------------------
ADDR_EMB        DEC     256
ADDR_WQ         DEC     272
ADDR_WK         DEC     288
ADDR_WV         DEC     304
ADDR_W1         DEC     320
ADDR_W2         DEC     336
ADDR_WO         DEC     352
ADDR_LQ         DEC     368
ADDR_LK         DEC     384
ADDR_LV         DEC     400
ADDR_L1         DEC     416
ADDR_L2         DEC     432
ADDR_LO         DEC     448
ADDR_X          DEC     480
ADDR_Q          DEC     488
ADDR_K          DEC     496
ADDR_V          DEC     504
ADDR_S          DEC     512
ADDR_A          DEC     516
ADDR_O          DEC     520
ADDR_H          DEC     528
ADDR_H2         DEC     536
ADDR_LOGITS     DEC     544
ADDR_G_LOGITS   DEC     553
ADDR_G_H2       DEC     561
ADDR_G_H        DEC     569
ADDR_G_O        DEC     577
ADDR_G_V        DEC     585
ADDR_G_S        DEC     593
ADDR_G_A        DEC     597
ADDR_G_Q        DEC     601
ADDR_G_K        DEC     609
ADDR_G_X        DEC     617
ADDR_G_WQ       DEC     625
ADDR_G_WK       DEC     641
ADDR_G_WV       DEC     657
ADDR_G_W1       DEC     673
ADDR_G_W2       DEC     689
ADDR_G_WO       DEC     705
ADDR_GATHER     DEC     721
ADDR_TOKENS     DEC     464
ADDR_TARGET_OH  DEC     466

; ------------------------------------------------------------------
; SCALARS / CONSTANTS (data words).
; Ternary -1 is ones' complement 77776 octal. +0 is 0. +1 is 1.
; ------------------------------------------------------------------
VAL_ZERO        OCT     0
VAL_ONE         OCT     1
VAL_NEG1        OCT     77776           ; -1 (ones' complement)
VAL_POS8        DEC     8
VAL_NEG8        OCT     77767           ; -8 (ones' complement)
VAL_DEAD        DEC     16
VAL_DEADP1      DEC     17
VAL_LR          DEC     1               ; SGD step (integer)
VAL_BUDGET      DEC     1000            ; kernel cycle budget reload
VAL_ZBUF        OCT     10000           ; $1000 (kernel input buffer)
VAL_WBUF        OCT     20000           ; $2000 (kernel weight buffer)
VAL_ITER_N      DEC     3               ; training iterations

; ------------------------------------------------------------------
; DATA TABLES. Ternary weights are -1/0/1 (OCT 77776 / 0 / 1).
; Latent weights are ternary * 64 (DEC 64 / 0 / -64 as OCT 77677).
; Embedding: token 0 = [1,0,-1,1], token 1 = [0,1,1,-1],
;            token 2 = [-1,1,0,1], token 3 = [1,-1,1,0].
; Tokens: [0, 1]. Target: token 2 (one-hot [0,0,1,0]).
; ------------------------------------------------------------------
EMB_TBL:
    OCT     1
    OCT     0
    OCT     77776
    OCT     1
    OCT     0
    OCT     1
    OCT     1
    OCT     77776
    OCT     77776
    OCT     1
    OCT     0
    OCT     1
    OCT     1
    OCT     77776
    OCT     1
    OCT     0
; WQ: rows (output dim) x cols (input dim). Mostly identity + spread.
WQ_TBL:
    OCT     1
    OCT     0
    OCT     0
    OCT     77776
    OCT     0
    OCT     1
    OCT     77776
    OCT     0
    OCT     0
    OCT     77776
    OCT     1
    OCT     0
    OCT     77776
    OCT     0
    OCT     0
    OCT     1
WK_TBL:
    OCT     1
    OCT     77776
    OCT     0
    OCT     0
    OCT     77776
    OCT     1
    OCT     0
    OCT     0
    OCT     0
    OCT     0
    OCT     1
    OCT     77776
    OCT     0
    OCT     0
    OCT     77776
    OCT     1
WV_TBL:
    OCT     1
    OCT     0
    OCT     77776
    OCT     0
    OCT     0
    OCT     1
    OCT     0
    OCT     77776
    OCT     77776
    OCT     0
    OCT     1
    OCT     0
    OCT     0
    OCT     77776
    OCT     0
    OCT     1
W1_TBL:
    OCT     1
    OCT     0
    OCT     0
    OCT     0
    OCT     0
    OCT     1
    OCT     0
    OCT     0
    OCT     0
    OCT     0
    OCT     1
    OCT     77776
    OCT     0
    OCT     0
    OCT     77776
    OCT     1
W2_TBL:
    OCT     1
    OCT     77776
    OCT     0
    OCT     0
    OCT     77776
    OCT     1
    OCT     0
    OCT     0
    OCT     0
    OCT     0
    OCT     1
    OCT     0
    OCT     0
    OCT     0
    OCT     0
    OCT     1
WO_TBL:
    OCT     1
    OCT     0
    OCT     77776
    OCT     0
    OCT     0
    OCT     1
    OCT     0
    OCT     77776
    OCT     77776
    OCT     0
    OCT     1
    OCT     0
    OCT     0
    OCT     77776
    OCT     0
    OCT     1
; Latent weights: ternary * 64. -64 is OCT 77677 (ones' complement).
LQ_TBL:
    DEC     64
    DEC     0
    DEC     0
    OCT     77677
    DEC     0
    DEC     64
    OCT     77677
    DEC     0
    DEC     0
    OCT     77677
    DEC     64
    DEC     0
    OCT     77677
    DEC     0
    DEC     0
    DEC     64
LK_TBL:
    DEC     64
    OCT     77677
    DEC     0
    DEC     0
    OCT     77677
    DEC     64
    DEC     0
    DEC     0
    DEC     0
    DEC     0
    DEC     64
    OCT     77677
    DEC     0
    DEC     0
    OCT     77677
    DEC     64
LV_TBL:
    DEC     64
    DEC     0
    OCT     77677
    DEC     0
    DEC     0
    DEC     64
    DEC     0
    OCT     77677
    OCT     77677
    DEC     0
    DEC     64
    DEC     0
    DEC     0
    OCT     77677
    DEC     0
    DEC     64
L1_TBL:
    DEC     64
    DEC     0
    DEC     0
    DEC     0
    DEC     0
    DEC     64
    DEC     0
    DEC     0
    DEC     0
    DEC     0
    DEC     64
    OCT     77677
    DEC     0
    DEC     0
    OCT     77677
    DEC     64
L2_TBL:
    DEC     64
    OCT     77677
    DEC     0
    DEC     0
    OCT     77677
    DEC     64
    DEC     0
    DEC     0
    DEC     0
    DEC     0
    DEC     64
    DEC     0
    DEC     0
    DEC     0
    DEC     0
    DEC     64
LO_TBL:
    DEC     64
    DEC     0
    OCT     77677
    DEC     0
    DEC     0
    DEC     64
    DEC     0
    OCT     77677
    OCT     77677
    DEC     0
    DEC     64
    DEC     0
    DEC     0
    OCT     77677
    DEC     0
    DEC     64
TOKENS_TBL:
    DEC     0
    DEC     1
TARGET_OH_TBL:
    DEC     0
    DEC     0
    DEC     1
    DEC     0
TARGET_TOK_VAL:
    DEC     2

; ------------------------------------------------------------------
; ENTRY + TRAINING LOOP.
; ------------------------------------------------------------------
START:
    ; Initialize scalars from the constant cells.
    CA      VAL_ZERO
    TS      ZERO_VAL
    CA      VAL_ONE
    TS      ONE_CONSTANT
    CA      VAL_POS8
    TS      POS8
    CA      VAL_NEG8
    TS      NEG8
    CA      VAL_DEAD
    TS      DEAD_VAL
    CA      VAL_DEADP1
    TS      DEADP1
    CA      VAL_LR
    TS      LR_VAL
    CA      VAL_BUDGET
    TS      BUDGET_VAL
    CA      VAL_ZBUF
    TS      PTR_ZBUF
    CA      VAL_WBUF
    TS      PTR_WBUF
    ; Copy data tables into their fixed locations.
    TC      COPY_TABLES
    ; (COPY_TABLES returns via START_AFTER_COPY below.)
TRAIN_LOOP:
    ; Forward pass (ends by TCing to BWD_START).
    TC      FWD_START
    ; Backward pass returns here (BWD ends with TC TRAIN_AFTER_BWD).
TRAIN_AFTER_BWD:
    ; ITER += 1. If ITER < ITER_N, run another iteration.
    CA      ONE_CONSTANT
    ADS     ITER
    CA      ITER
    TS      TMP0
    CS      VAL_ITER_N
    ADS     TMP0                ; TMP0 = ITER - ITER_N
    CA      TMP0
    EXTEND
    BM      TRAIN_LOOP          ; ITER < ITER_N: loop again
    ; Done. HALT (return to executive). In a real AGC this would
    ; TC back to the executive; here it is a clean stop.
    TC      HALT
HALT:
    NOOP
    TC      HALT                ; spin (no AGC HALT instruction)

; ------------------------------------------------------------------
; COPY_TABLES -- copy the _TBL data into the fixed table locations.
; (Tables are assembled at their _TBL labels; copy to EMB/WQ/...).
; ------------------------------------------------------------------
COPY_TABLES:
    ; EMB <- EMB_TBL (16 words).
    CA      ADDR_EMB
    TS      COPY_DST
    CA      EMB_TBL_ADDR
    TS      COPY_SRC
    CA      SIXTEEN
    TS      COPY_CNT
    TC      COPY_WORDS
    ; WQ <- WQ_TBL.
    CA      ADDR_WQ
    TS      COPY_DST
    CA      WQ_TBL_ADDR
    TS      COPY_SRC
    CA      SIXTEEN
    TS      COPY_CNT
    TC      COPY_WORDS
    ; WK <- WK_TBL.
    CA      ADDR_WK
    TS      COPY_DST
    CA      WK_TBL_ADDR
    TS      COPY_SRC
    CA      SIXTEEN
    TS      COPY_CNT
    TC      COPY_WORDS
    ; WV <- WV_TBL.
    CA      ADDR_WV
    TS      COPY_DST
    CA      WV_TBL_ADDR
    TS      COPY_SRC
    CA      SIXTEEN
    TS      COPY_CNT
    TC      COPY_WORDS
    ; W1 <- W1_TBL.
    CA      ADDR_W1
    TS      COPY_DST
    CA      W1_TBL_ADDR
    TS      COPY_SRC
    CA      SIXTEEN
    TS      COPY_CNT
    TC      COPY_WORDS
    ; W2 <- W2_TBL.
    CA      ADDR_W2
    TS      COPY_DST
    CA      W2_TBL_ADDR
    TS      COPY_SRC
    CA      SIXTEEN
    TS      COPY_CNT
    TC      COPY_WORDS
    ; WO <- WO_TBL.
    CA      ADDR_WO
    TS      COPY_DST
    CA      WO_TBL_ADDR
    TS      COPY_SRC
    CA      SIXTEEN
    TS      COPY_CNT
    TC      COPY_WORDS
    ; LQ <- LQ_TBL.
    CA      ADDR_LQ
    TS      COPY_DST
    CA      LQ_TBL_ADDR
    TS      COPY_SRC
    CA      SIXTEEN
    TS      COPY_CNT
    TC      COPY_WORDS
    ; LK <- LK_TBL.
    CA      ADDR_LK
    TS      COPY_DST
    CA      LK_TBL_ADDR
    TS      COPY_SRC
    CA      SIXTEEN
    TS      COPY_CNT
    TC      COPY_WORDS
    ; LV <- LV_TBL.
    CA      ADDR_LV
    TS      COPY_DST
    CA      LV_TBL_ADDR
    TS      COPY_SRC
    CA      SIXTEEN
    TS      COPY_CNT
    TC      COPY_WORDS
    ; L1 <- L1_TBL.
    CA      ADDR_L1
    TS      COPY_DST
    CA      L1_TBL_ADDR
    TS      COPY_SRC
    CA      SIXTEEN
    TS      COPY_CNT
    TC      COPY_WORDS
    ; L2 <- L2_TBL.
    CA      ADDR_L2
    TS      COPY_DST
    CA      L2_TBL_ADDR
    TS      COPY_SRC
    CA      SIXTEEN
    TS      COPY_CNT
    TC      COPY_WORDS
    ; LO <- LO_TBL.
    CA      ADDR_LO
    TS      COPY_DST
    CA      LO_TBL_ADDR
    TS      COPY_SRC
    CA      SIXTEEN
    TS      COPY_CNT
    TC      COPY_WORDS
    ; TOKENS <- TOKENS_TBL (2 words).
    CA      ADDR_TOKENS
    TS      COPY_DST
    CA      TOKENS_TBL_ADDR
    TS      COPY_SRC
    CA      TWO
    TS      COPY_CNT
    TC      COPY_WORDS
    ; TARGET_OH <- TARGET_OH_TBL (4 words).
    CA      ADDR_TARGET_OH
    TS      COPY_DST
    CA      TARGET_OH_TBL_ADDR
    TS      COPY_SRC
    CA      FOUR
    TS      COPY_CNT
    TC      COPY_WORDS
    ; TARGET_TOK <- TARGET_TOK_VAL (1 word).
    CA      TARGET_TOK_VAL
    TS      TARGET_TOK
    ; Return to START (via HELPER_RET-style: COPY_TABLES is called
    ; with TC, so it returns through the caller's continuation).
    ; START calls COPY_TABLES with TC; COPY_TABLES returns by
    ; TCing to the label after the call (START_AFTER_COPY).
    TC      START_AFTER_COPY
START_AFTER_COPY:
    ; (Placeholder: START falls through here after COPY_TABLES.
    ;  Kept as an explicit label so the flow is clear.)
    CA      ZERO_VAL
    TS      ITER
    TC      TRAIN_LOOP

; COPY_WORDS -- copy COPY_CNT words from [COPY_SRC] to [COPY_DST].
; Advances both pointers. Called with TC (standard AGC return):
; the caller TCs here, Q holds the return address, and COPY_WORDS
; returns with TC Q. COPY_WORDS is a leaf (no nested TC calls).
COPY_WORDS:
    INDEX   COPY_SRC
    CA      0
    INDEX   COPY_DST
    TS      0
    CA      ONE_CONSTANT
    ADS     COPY_SRC
    CA      ONE_CONSTANT
    ADS     COPY_DST
    CCS     COPY_CNT
    TC      COPY_WORDS
    TC      Q
    TC      Q

; Small count constants (data words).
TWO             DEC     2
FOUR            DEC     4
SIXTEEN         DEC     16
; Table address cells (the _TBL labels' addresses). These are filled
; by the assembler: each holds the address of its table.
EMB_TBL_ADDR    EMB_TBL
WQ_TBL_ADDR     WQ_TBL
WK_TBL_ADDR     WK_TBL
WV_TBL_ADDR     WV_TBL
W1_TBL_ADDR     W1_TBL
W2_TBL_ADDR     W2_TBL
WO_TBL_ADDR     WO_TBL
LQ_TBL_ADDR     LQ_TBL
LK_TBL_ADDR     LK_TBL
LV_TBL_ADDR     LV_TBL
L1_TBL_ADDR     L1_TBL
L2_TBL_ADDR     L2_TBL
LO_TBL_ADDR     LO_TBL
TOKENS_TBL_ADDR TOKENS_TBL
TARGET_OH_TBL_ADDR TARGET_OH_TBL

; ------------------------------------------------------------------
; CALL_TERNARY -- stage operands into the kernel's fixed buffers
; and invoke APOLLO_TERNARY_REDUCTION.
; Inputs: ARG_X (input vector base), ARG_W (weight row base),
;         ARG_N (length). The caller sets RETURN_SLOT to the
;         continuation ID before TCing here. On return (via
;         RESTORE_EXECUTIVE dispatcher) ACCUM_REG_L holds the dot
;         product; the caller's continuation reads it.
; ------------------------------------------------------------------
CALL_TERNARY:
    ; Copy ARG_X (ARG_N words) to Z buffer ($1000).
    CA      ARG_X
    TS      COPY_SRC
    CA      PTR_ZBUF
    TS      COPY_DST
    CA      ARG_N
    TS      COPY_CNT
    TC      COPY_WORDS_X
COPY_WORDS_X:
    INDEX   COPY_SRC
    CA      0
    INDEX   COPY_DST
    TS      0
    CA      ONE_CONSTANT
    ADS     COPY_SRC
    CA      ONE_CONSTANT
    ADS     COPY_DST
    CCS     COPY_CNT
    TC      COPY_WORDS_X
    TC      COPY_W_STAGE
    TC      COPY_W_STAGE
COPY_W_STAGE:
    ; Copy ARG_W (ARG_N words) to W buffer ($2000).
    CA      ARG_W
    TS      COPY_SRC
    CA      PTR_WBUF
    TS      COPY_DST
    CA      ARG_N
    TS      COPY_CNT
    TC      COPY_WORDS_W
COPY_WORDS_W:
    INDEX   COPY_SRC
    CA      0
    INDEX   COPY_DST
    TS      0
    CA      ONE_CONSTANT
    ADS     COPY_SRC
    CA      ONE_CONSTANT
    ADS     COPY_DST
    CCS     COPY_CNT
    TC      COPY_WORDS_W
    TC      KERNEL_START
    TC      KERNEL_START
KERNEL_START:
    CA      ARG_N
    TS      VECTOR_COUNT
    CA      BUDGET_VAL
    TS      CYCLE_BUDGET
    TC      APOLLO_TERNARY_REDUCTION

; ------------------------------------------------------------------
; RESTORE_EXECUTIVE -- the kernel's fixed return point.
; Dispatches through RET_TABLE using RETURN_SLOT.
; ------------------------------------------------------------------
RESTORE_EXECUTIVE:
    INDEX   RETURN_SLOT
    TC      RET_TABLE
RET_TABLE:
    TC      FWD_DOT_CONT        ; RETURN_SLOT 0: forward dot-loop cont.
    TC      BWD_DOT_CONT        ; RETURN_SLOT 1: backward dot-loop cont.
    TC      FWD_DOT_CONT        ; (spare, same as 0)
    TC      BWD_DOT_CONT        ; (spare, same as 1)
