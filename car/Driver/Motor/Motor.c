#include "Motor.h"


void Motor_init(){
	Gpio_initPin(&DDRB, OUTPUT, 1);
}

void Motor_speedMode(uint16_t data){
	MOTOR_ICR = (250000 / 250) - 1;
	if(data == 0) MOTOR_OCR = 0;
	else MOTOR_OCR = MOTOR_ICR / data;
}
