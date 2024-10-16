#include "Service.h"

static uint8_t lastMotorState = FAST;

void Service_MotorData(){
    uint8_t MotorUARTState;
    uint8_t MotorHCState;

    // 상태 가져오기
    MotorUARTState = Model_getMotorUARTStateData();  // UART로부터 현재 상태 가져옴
    MotorHCState = Model_getMotorHCStateData();      // 초음파 센서로부터 현재 상태 가져옴

    // 1. 초음파 상태가 STOP이면 무조건 모터를 멈춤
    if (MotorHCState == STOP || MotorUARTState == STOP) {
        Motor_speedMode(FAST);    // 모터를 멈춤
        Motor_Left();
        lastMotorState = STOP;    // 마지막 상태 업데이트
        UART0_sendString("Stop\n");
        return;                   // 더 이상 처리하지 않고 리턴
    }

    // 2. UART 명령이 있는 경우 우선 적용
    if (MotorUARTState != lastMotorState) {
        if (MotorUARTState == STOP) {
            Motor_speedMode(STOP);  // UART로 STOP 명령이 오면 모터를 멈춤
            Motor_Stop();
            lastMotorState = STOP;
            UART0_sendString("Stop\n");
            return;
        } 
        else if (MotorUARTState == FAST) {
            Motor_speedMode(FAST);  // UART로 FAST 명령이 오면 모터를 빠르게 작동
            Motor_Go();  
            UART0_sendString("Fast\n");
            lastMotorState = FAST;
        }
    }

    // 3. 기본 상태는 FAST로 설정
    if (MotorHCState != STOP && MotorUARTState == FAST) {  // 초음파가 STOP이 아니고, UART 명령이 FAST일 때만
        Motor_speedMode(FAST);      // 기본 상태로 모터를 FAST로 작동
        Motor_Go();
        UART0_sendString("Fast\n");
        lastMotorState = FAST;      // 마지막 상태 업데이트
    }
}


void Service_execute(){
    Service_MotorData();
}