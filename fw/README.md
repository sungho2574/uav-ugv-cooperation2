# 업로드 방법

1. 플래시 (부트로더 모드 필요 없음!)

   ```bash
   cfloader flash aideck_gap8_hj.bin deck-bcAI:gap8-fw -w radio://0/80/2M/E7E7E7E7E7
   ```

2. 노트북을 같은 와이파이에 연결

3. 크플을 cfclinet에 연결해서 와이파이 연결됐는지 확인

   ```bash
   CPX: ESP32: I (57258) WIFI: rssi: -30
   CPX: ESP32: I (57258) WIFI: 11b: 1, 11g: 1, 11n: 1, lr: 0
   CPX: WiFi connected to ip: 172.20.10.13
   CPX: GAP8: Wifi connected (172.20.10.13)
   ```

4. 노트북에서 위에 나온 아이피 주소로 연결

   ```bash
    # cd examples/other/wifi-img-streamer-multiple/single-viewer.py
    python single-viewer.py 172.20.10.13
   ```
