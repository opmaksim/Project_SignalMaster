#include "apMain.h"

ISR(USART_RX_vect){
    UART0_ISR_Process();
}

void apMain_init(){
    Listener_init();
    TIM0_init();
    TIM1_init();
    sei();
    Model_setMotorUARTStateData(FAST);
}

void apMain_execute(){
    Listener_EventCheck();
    Service_execute();
}