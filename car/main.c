#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

#include "Periph/GPIO/GPIO.h"
#include "Periph/TIM/TIM.h"
#include "Driver/Motor/Motor.h"
#include "Periph/UART0/UART0.h"


volatile int milisec = 0;
volatile int sec = 0;
ISR(TIMER0_COMPA_vect) {
    milisec++;
    if (milisec >= 1000) {  // 1초 경과
        milisec = 0;
        sec = (sec + 1) % 100;
    }
}

ISR(USART_RX_vect){
    UART0_ISR_Process();
}

#define LED_PIN PB5  // 아두이노 우노 내장 LED는 PB5 (13번 핀)

void processCommand(char *command) {
    if (strcmp((uint8_t*)command, 1) == 0) {
        PORTB |= (1 << LED_PIN);  // LED 켜기
        UART0_sendString("LED ON\n");
    } else if (strcmp(command, "2\n") == 0) {
        PORTB &= ~(1 << LED_PIN); // LED 끄기
        UART0_sendString("LED OFF\n");
    } else {
        UART0_sendString("Unknown command\n");
    }
}


int main(void) {
    sei();
    USART_Init();
    TIM0_init();
    TIM1_init();
    Motor_init();
    DDRB |= (1 << LED_PIN);  // LED_PIN(13번 핀)을 출력으로 설정
    while (1) {
        if (UART0_getRxFlag()) {
            // 수신된 데이터를 처리
            char* receivedData = (char*)UART0_readRxBuff();
            processCommand(receivedData);  // 명령 처리
            
            UART0_clearRxFlag();  // 플래그 리셋
        }
    }
    return 0;
}
