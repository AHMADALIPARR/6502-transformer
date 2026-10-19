; ==================================================================
; AGC TRANSFORMER -- MATH HELPERS
; Small, leaf routines. Each is called by setting HELPER_RET to a
; continuation label, then TC HELPER. The helper ends with
; INDEX HELPER_RET / TC 0 (indirect jump to the continuation).
; Helpers are LEAF routines: they never TC to another routine
; (so they never clobber a return address). No hardware MP/DV is
; used; multiply/divide are repeated add/subtract, per the repo's
; stated constraint.
;
; STATUS: WRITTEN, NOT EXECUTED. No AGC emulator here. Do NOT
; claim these run. Ones' complement: values are kept small so
; products stay inside 15 bits; -0 is avoided by construction.
; Memory map lives in main.agc; labels below are defined there.
; ==================================================================

; ------------------------------------------------------------------
; MUL8: MUL_R = MUL_A * MUL_B (signed, small values).
; Method: absolute values, repeated addition, fix sign.
; Inputs: MUL_A, MUL_B. Output: MUL_R.
; Clobbers: MUL_A_ABS, MUL_B_ABS, MUL_SIGN, TMP0.
; ------------------------------------------------------------------
MUL8:
    CA      ZERO_VAL
    TS      MUL_R
    ; |A|
    CA      MUL_A
    EXTEND
    BM      MUL_A_NEG
    TS      MUL_A_ABS
    TC      MUL_A_DONE
MUL_A_NEG:
    CS      MUL_A
    TS      MUL_A_ABS
MUL_A_DONE:
    ; |B|
    CA      MUL_B
    EXTEND
    BM      MUL_B_NEG
    TS      MUL_B_ABS
    TC      MUL_B_DONE
MUL_B_NEG:
    CS      MUL_B
    TS      MUL_B_ABS
MUL_B_DONE:
    ; If |B| == 0, product is 0. Done.
    CA      MUL_B_ABS
    EXTEND
    BZ      MUL_RETURN
    TS      TMP0                ; TMP0 = |B| (loop counter)
    ; Sign: negative if exactly one of A,B is negative.
    CA      ZERO_VAL
    TS      MUL_SIGN
    CA      MUL_A
    EXTEND
    BM      MUL_A_IS_NEG
    TC      MUL_CHK_B
MUL_A_IS_NEG:
    CA      ONE_CONSTANT
    TS      MUL_SIGN
MUL_CHK_B:
    CA      MUL_B
    EXTEND
    BM      MUL_B_IS_NEG
    TC      MUL_LOOP
MUL_B_IS_NEG:
    ; Toggle MUL_SIGN.
    CA      MUL_SIGN
    EXTEND
    BZ      MUL_SET_NEG
    CA      ZERO_VAL            ; was 1 -> 0
    TS      MUL_SIGN
    TC      MUL_LOOP
MUL_SET_NEG:
    CA      ONE_CONSTANT        ; was 0 -> 1
    TS      MUL_SIGN
MUL_LOOP:
    CA      MUL_A_ABS
    ADS     MUL_R               ; MUL_R += |A|
    CCS     TMP0                ; TMP0 -= 1
    TC      MUL_LOOP            ; TMP0 > 0: loop
    TC      MUL_SIGN_FIX        ; TMP0 == 0: fix sign
    TC      MUL_SIGN_FIX        ; TMP0 < 0: (unreached) fix sign
MUL_SIGN_FIX:
    CA      MUL_SIGN
    EXTEND
    BZ      MUL_RETURN
    CS      MUL_R
    TS      MUL_R
MUL_RETURN:
    INDEX   HELPER_RET
    TC      0

; ------------------------------------------------------------------
; DIV: DIV_Q = DIV_N / DIV_D (integer), DIV_R = remainder.
; Inputs: DIV_N (>= 0), DIV_D (> 0). Caller must ensure DIV_D > 0
; (divide by zero loops forever). Method: repeated subtraction.
; Clobbers: TMP0.
; ------------------------------------------------------------------
DIV:
    CA      ZERO_VAL
    TS      DIV_Q
    CA      DIV_N
    TS      DIV_R               ; DIV_R = running remainder
DIV_LOOP:
    ; TMP0 = DIV_R - DIV_D. If < 0, done.
    CA      DIV_R
    TS      TMP0
    CS      DIV_D
    ADS     TMP0                ; TMP0 = DIV_R - DIV_D
    CA      TMP0
    EXTEND
    BM      DIV_DONE
    TS      DIV_R               ; DIV_R = TMP0
    CA      ONE_CONSTANT
    ADS     DIV_Q               ; DIV_Q += 1
    TC      DIV_LOOP
DIV_DONE:
    INDEX   HELPER_RET
    TC      0

; ------------------------------------------------------------------
; EXP_LOOKUP: EXP_OUT = exp(EXP_IN) * 128, EXP_IN in [-8, 0].
; EXP_IN is clamped to [-8, 0]. Table (scale 128):
;   x=0:128, -1:47, -2:17, -3:6, -4:2, -5:1, -6:0, -7:0, -8:0.
; Softmax subtracts the row max first, so EXP_IN is always <= 0.
; Clobbers: TMP0.
; ------------------------------------------------------------------
EXP_LOOKUP:
    CA      EXP_IN
    EXTEND
    BM      EXP_NEG             ; EXP_IN < 0: check lower bound
    CA      ZERO_VAL            ; EXP_IN >= 0: use x = 0
    TS      EXP_IDX
    TC      EXP_FETCH
EXP_NEG:
    ; If EXP_IN < -8, clamp to -8. Else EXP_IDX = -EXP_IN.
    CA      EXP_IN
    TS      TMP0
    CA      POS8
    ADS     TMP0                ; TMP0 = EXP_IN + 8
    CA      TMP0
    EXTEND
    BM      EXP_CLAMP8          ; EXP_IN + 8 < 0 -> EXP_IN < -8
    CS      EXP_IN              ; EXP_IDX = -EXP_IN (0..8)
    TS      EXP_IDX
    TC      EXP_FETCH
EXP_CLAMP8:
    CA      POS8
    TS      EXP_IDX             ; EXP_IDX = 8 (x = -8)
EXP_FETCH:
    INDEX   EXP_IDX
    CA      EXP_TBL
    TS      EXP_OUT
    INDEX   HELPER_RET
    TC      0
EXP_TBL:
    DEC     128                 ; x = 0
    DEC     47                  ; x = -1
    DEC     17                  ; x = -2
    DEC     6                   ; x = -3
    DEC     2                   ; x = -4
    DEC     1                   ; x = -5
    DEC     0                   ; x = -6
    DEC     0                   ; x = -7
    DEC     0                   ; x = -8

; ------------------------------------------------------------------
; RELU: RELU_VAL = max(RELU_VAL, 0), in place.
; ------------------------------------------------------------------
RELU:
    CA      RELU_VAL
    EXTEND
    BM      RELU_NEG
    INDEX   HELPER_RET
    TC      0
RELU_NEG:
    CA      ZERO_VAL
    TS      RELU_VAL
    INDEX   HELPER_RET
    TC      0

; ------------------------------------------------------------------
; CLAMP8: CLAMP_OUT = clamp(CLAMP_IN, -8, 8).
; Clobbers: TMP0.
; ------------------------------------------------------------------
CLAMP8:
    CA      CLAMP_IN
    EXTEND
    BM      CLAMP_NEG
    ; CLAMP_IN >= 0. If > 8, clamp to 8.
    CA      CLAMP_IN
    TS      TMP0
    CS      POS8
    ADS     TMP0                ; TMP0 = CLAMP_IN - 8
    CA      TMP0
    EXTEND
    BM      CLAMP_POS_OK        ; CLAMP_IN < 8: keep
    CA      POS8
    TS      CLAMP_OUT
    INDEX   HELPER_RET
    TC      0
CLAMP_POS_OK:
    CA      CLAMP_IN
    TS      CLAMP_OUT
    INDEX   HELPER_RET
    TC      0
CLAMP_NEG:
    ; CLAMP_IN < 0. If < -8, clamp to -8.
    CA      CLAMP_IN
    TS      TMP0
    CA      POS8
    ADS     TMP0                ; TMP0 = CLAMP_IN + 8
    CA      TMP0
    EXTEND
    BM      CLAMP_NEG8          ; CLAMP_IN < -8: clamp
    CA      CLAMP_IN
    TS      CLAMP_OUT
    INDEX   HELPER_RET
    TC      0
CLAMP_NEG8:
    CS      POS8                ; A = -8
    TS      CLAMP_OUT
    INDEX   HELPER_RET
    TC      0

; ------------------------------------------------------------------
; QUANTIZE: QUANT_OUT = dead-zone quantize(QUANT_IN).
; |latent| > 16 -> sign (1 or -1). |latent| <= 16 -> 0.
; The dead zone is what makes ternary weights sparse (0s survive).
; Clobbers: TMP0.
; ------------------------------------------------------------------
QUANTIZE:
    CA      QUANT_IN
    EXTEND
    BM      QUANT_NEG
    ; QUANT_IN >= 0. If QUANT_IN > 16 -> 1, else 0.
    ; Test QUANT_IN - 17 >= 0 (strictly greater than 16).
    CA      QUANT_IN
    TS      TMP0
    CS      DEADP1
    ADS     TMP0                ; TMP0 = QUANT_IN - 17
    CA      TMP0
    EXTEND
    BM      QUANT_ZERO          ; QUANT_IN <= 16 -> 0
    CA      ONE_CONSTANT
    TS      QUANT_OUT
    INDEX   HELPER_RET
    TC      0
QUANT_ZERO:
    CA      ZERO_VAL
    TS      QUANT_OUT
    INDEX   HELPER_RET
    TC      0
QUANT_NEG:
    ; QUANT_IN < 0. If QUANT_IN < -16 -> -1, else 0.
    ; Test QUANT_IN + 17 < 0 (strictly less than -16).
    CA      QUANT_IN
    TS      TMP0
    CA      DEADP1
    ADS     TMP0                ; TMP0 = QUANT_IN + 17
    CA      TMP0
    EXTEND
    BM      QUANT_MINUS         ; QUANT_IN <= -17 -> -1
    CA      ZERO_VAL
    TS      QUANT_OUT
    INDEX   HELPER_RET
    TC      0
QUANT_MINUS:
    CS      ONE_CONSTANT        ; A = -1
    TS      QUANT_OUT
    INDEX   HELPER_RET
    TC      0

; ------------------------------------------------------------------
; SIGN: SIGN_OUT = sign(SIGN_IN): -1, 0, or 1.
; ------------------------------------------------------------------
SIGN:
    CA      SIGN_IN
    EXTEND
    BZ      SIGN_ZERO
    EXTEND
    BM      SIGN_NEG
    CA      ONE_CONSTANT
    TS      SIGN_OUT
    INDEX   HELPER_RET
    TC      0
SIGN_ZERO:
    CA      ZERO_VAL
    TS      SIGN_OUT
    INDEX   HELPER_RET
    TC      0
SIGN_NEG:
    CS      ONE_CONSTANT
    TS      SIGN_OUT
    INDEX   HELPER_RET
    TC      0

; ------------------------------------------------------------------
; ABS: ABS_OUT = |ABS_IN|.
; ------------------------------------------------------------------
ABS:
    CA      ABS_IN
    EXTEND
    BM      ABS_NEG
    TS      ABS_OUT
    INDEX   HELPER_RET
    TC      0
ABS_NEG:
    CS      ABS_IN
    TS      ABS_OUT
    INDEX   HELPER_RET
    TC      0
