#ifndef UART0_H_
#define UART0_H_
#define F_CPU 16000000UL

#define FOSC 16000000UL // Clock Speed
#define BAUD 115200
#define MYUBRR FOSC/(BAUD*16)-1

#include <avr/io.h>
#include <util/delay.h>
void USART_Init();
void USART_Transmit(unsigned char data);
unsigned char USART_Receive(void);
#endif