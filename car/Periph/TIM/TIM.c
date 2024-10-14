#include "TIM.h"


// void TIM0_init(){
//     TCCR0B |= ((0<<CS02) | (1<<CS01) | (1<<CS00));
//     //CTC Mode
//     TCCR0B |= ((0<<WGM02));
//     TCCR0A |= ((1<<WGM01) | (0<<WGM00));
//     TCCR0A &= ~((1<<COM0A1) | (1<<COM0A0));
//     OCR0A = 250-1;
//     TIMSK0 |= (1 << OCIE0A);
// }

void TIM0_init(){
    // Timer0 설정 (Normal 모드, Prescaler 64)
    TCCR0A = 0x00;
    TCCR0B = (1 << CS01) | (1 << CS00);  // Prescaler 64
    TCNT0 = 0;  // 타이머 초기화
}

void TIM1_init(){
    TCCR1A |= ((1<<WGM11) | (0<<WGM10)); 
    TCCR1B |= ((1<<WGM13) | (1<<WGM12));
    TCCR1B |= ((0<<CS12) | (1<<CS11) | (1<<CS10));
    TCCR1A |= (1<<COM1A1) | (0<<COM1A0);
}
