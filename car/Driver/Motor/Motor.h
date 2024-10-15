#ifndef MOTOR_H_
#define MOTOR_H_
#define F_CPU 16000000UL
#include <avr/io.h>
#include <util/delay.h>
#include "../../Periph/GPIO/GPIO.h"
#define MOTOR_ICR		ICR1
#define MOTOR_OCR		OCR1A


void Motor_init();
void Motor_speedMode(uint16_t data);
#endif