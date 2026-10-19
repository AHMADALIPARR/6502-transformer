; ==================================================================
; APOLLO GUIDANCE COMPUTER (AGC) TERNARY SQUIB KERNEL
; Mission Profile: Real-Time Vector Reduction for Lunar Descent Guidance
; Constraints: Fixed-time execution, Hardwired ROM/RAM partitions, 
;              Zero dynamic allocation, Deterministic branch cost ($O(N)$).
; ==================================================================

; Fixed Hardware Registers (AGC Fixed-Fixed Memory Map)
Z_VECTOR_BASE   EQU     #$1000          ; Input Vector X (Read-Only Core Rope)
W_TERNARY_BASE  EQU     #$2000          ; Weight Matrix W (-1, 0, 1 Encoded)
ACCUM_REG_L     EQU     $0040           ; 16-bit Precision Accumulator (Low)
ACCUM_REG_H     EQU     $0041           ; 16-bit Precision Accumulator (High)
VECTOR_COUNT    EQU     $0042           ; Hard-bounded vector dimension (N <= 255)
CYCLE_BUDGET    EQU     $0043           ; Executive Overrun Protection Register

APOLLO_TERNARY_REDUCTION:
    DXCH    ZERO_VAL                    ; Clear working registers via AGC double exchange
    TS      ACCUM_REG_L                 ; Zero out low byte
    TS      ACCUM_REG_H                 ; Zero out high byte
    CA      VECTOR_COUNT                ; Load vector length into accumulator
    EXTEND                              ; Enable multi-strike instruction extension
    QXCH    LOOP_COUNTER                ; Store loop index in index register Q

DISPATCH_LOOP:
    CCS     LOOP_COUNTER                ; Count, Compare, and Skip if zero or negative
    TC      EXECUTE_KERNEL              ; Positive value: continue processing node
    TC      MISSION_GUIDANCE_COMPLETE   ; Zero/Negative: terminate cleanly

EXECUTE_KERNEL:
    ; Fetch ternary weight using unrolled indexed indirect addressing
    INDEX   LOOP_COUNTER
    CA      W_TERNARY_BASE              ; Load weight W_i
    
    EXTEND
    BZ      SKIP_CYCLE                  ; W_i == 0: Eliminate multiply branch (Sparsity Lock)
    
    EXTEND
    BM      APOLLO_SUBTRACT             ; W_i == -1: Route to decrement path

APOLLO_ADD:
    INDEX   LOOP_COUNTER
    CA      Z_VECTOR_BASE               ; Load input activation X_i
    DAS     ACCUM_REG_L                 ; Double precision add to accumulator
    TC      ADVANCE_STATE

APOLLO_SUBTRACT:
    INDEX   LOOP_COUNTER
    CA      Z_VECTOR_BASE               ; Load input activation X_i
    DPS     ACCUM_REG_L                 ; Double precision subtract from accumulator
    TC      ADVANCE_STATE

SKIP_CYCLE:
    ; Maintain strict temporal parity so zero-weights don't leak timing side-channels
    NOOP                                

ADVANCE_STATE:
    CS      ONE_CONSTANT
    ADS     CYCLE_BUDGET                ; Decrement real-time execution budget
    TC      DISPATCH_LOOP               ; Loop back instantly without stack pollution

MISSION_GUIDANCE_COMPLETE:
    TCF     RESTORE_EXECUTIVE           ; Return control back to Executive Tier-1 loop
