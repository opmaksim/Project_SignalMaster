#ifndef HC_SR04_H_
#define HC_SR04_H_
#define F_CPU 16000000UL
#include <avr/io.h>
#include <util/delay.h>
#include "../../Periph/GPIO/GPIO.h"
#define TRIG_PIN 2
#define ECHO_PIN 3
void ultrasonic_init();
void trigger_pulse();
uint16_t measure_distance();
#endif