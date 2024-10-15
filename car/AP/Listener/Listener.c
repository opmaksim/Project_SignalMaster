#include "Listener.h"
uint8_t distance;

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
    distance = measure_distance();
    uint8_t MotorHCState;
    MotorHCState = Model_getMotorHCStateData();
    switch (MotorHCState){
    case STOP:
        if (distance > 70){
            Model_setMotorHCStateData(FAST);
        }
        else if (distance <= 70 && distance > 30){
            Model_setMotorHCStateData(SLOW);
        }
        else if (distance <= 30){
            Model_setMotorHCStateData(STOP);
        }
        break;
    case FAST:   
        if (distance > 70){
            Model_setMotorHCStateData(FAST);
        }
        else if (distance <= 70 && distance > 30){
            Model_setMotorHCStateData(SLOW);
        }
        else if (distance <= 30){
            Model_setMotorHCStateData(STOP);
        }
        break;
    case SLOW:
        if (distance > 70){
            Model_setMotorHCStateData(FAST);
        }
        else if (distance <= 70 && distance > 30){
            Model_setMotorHCStateData(SLOW);
        }
        else if (distance <= 30){
            Model_setMotorHCStateData(STOP);
        }
        break;
    }
}

void Listener_MotorUartEvent(){
    distance = measure_distance();
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
        else if(!strcmp((uint8_t *)rxString, "SLOW\n")){
            Model_setMotorUARTStateData(SLOW);
            UART0_sendString("SLOW\n");
        }
        UART0_clearRxFlag();
    }
}
