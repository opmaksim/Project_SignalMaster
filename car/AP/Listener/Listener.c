#include "Listener.h"

void Listener_init(){
    Motor_init();
    USART_init();
    ultrasonic_init();
}

void Listener_EventCheck(){
    Listener_MotorHCEvent();
    Listener_MotorUartEvent();
}

void Listener_MotorHCEvent(){
    uint32_t distance1 = distance[0];  // 첫 번째 센서의 거리
    uint32_t distance2 = distance[1];  // 두 번째 센서의 거리
    uint32_t distance3 = distance[2];  // 세 번째 센서의 거리

    uint8_t MotorHCState;
    MotorHCState = Model_getMotorHCStateData();

    switch (MotorHCState){
    case STOP:
        // 하나라도 거리가 10보다 크면 FAST로 전환
        if (distance1 <= (unsigned long *)10 || distance2 <= (unsigned long *)10 || distance3 <= (unsigned long *)10) {
            Model_setMotorHCStateData(STOP);
        }
        else{
            Model_setMotorHCStateData(FAST);
        }   
    break;
    case FAST:   
        if (distance1 <= (unsigned long *)10 || distance2 <= (unsigned long *)10 || distance3 <= (unsigned long *)10) {
            Model_setMotorHCStateData(STOP);
        }
        else{
            Model_setMotorHCStateData(FAST);
        }   
        break;
    }
}



void Listener_MotorUartEvent(){
    uint8_t MotorHCState;
    MotorHCState = Model_getMotorUARTStateData();
    if(UART0_getRxFlag())
	{
		char* rxString = (char *)UART0_readRxBuff();
        if(!strcmp((uint8_t *)rxString, "STOP\n")){
            Model_setMotorUARTStateData(STOP);
            UART0_sendString("STOP\n");
        }
        else if(!strcmp((uint8_t *)rxString, "FAST\n")){
            Model_setMotorUARTStateData(FAST);
            UART0_sendString("FAST\n");
        }
        UART0_clearRxFlag();
    }
}