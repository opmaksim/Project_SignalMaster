#include "AP/apMain.h"

#include <stdio.h>  // 숫자를 문자열로 변환하기 위해 필요
#include <stdlib.h>
void send_distance_via_uart() {
    char buffer[50];  // 숫자를 문자열로 변환하여 저장할 버퍼
    for (uint8_t i = 0; i < 3; i++) {
        sprintf(buffer, "%d %lu", i, distance[i]);  // distance[i] 값을 문자열로 변환
        UART0_sendString("Sensor ");     // 센서 번호 출력
        UART0_sendString(": ");          // ": " 출력
        UART0_sendString(buffer);        // 변환된 거리 값 출력
        UART0_sendString(" cm\n");       // "cm" 단위 출력
    }
}


int main(void) {
    apMain_init();
    while (1) {
        apMain_execute();
        send_distance_via_uart();
    }
    return 0;
}
