#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

#include "Periph/GPIO/GPIO.h"
#include "Periph/TIM/TIM.h"
#include "Driver/Motor/Motor.h"


volatile int milisec = 0;
volatile int sec = 0;
ISR(TIMER0_COMPA_vect) {
    milisec++;
    if (milisec >= 1000) {  // 1초 경과
        milisec = 0;
        sec = (sec + 1) % 100;
    }
}

void Motor_speedControl(unsigned char command)
{
    switch (command) {
        case '1':
            // PWM 25% 설정
            Motor_speedMode(1);
            break;
        case '2':
            // PWM 50% 설정
            Motor_speedMode(2);
            break;
        case '3':
            // PWM 75% 설정
            Motor_speedMode(3);
            break;
        default:
            // 기본 모터 속도 0%
            Motor_speedMode(0);
            break;
    }
}

unsigned char receivedChar;
int main(void) {
    // 전역 인터럽트 허용
    sei();
    USART_Init();
    TIM0_init();
    TIM1_init();
    Motor_init();
    while (1) {
        if(sec > 1) sec = 0;
        // 시리얼 모니터로부터 데이터 수신
        receivedChar = USART_Receive();

        // 수신된 값에 따라 모터 속도 제어
        Motor_speedControl(receivedChar);
    }
    return 0;
}
