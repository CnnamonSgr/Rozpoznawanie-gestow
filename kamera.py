import cv2
import mediapipe as mp
import gesty

# Przypisuje zmiennej przechwyt kamery
kamera = cv2.VideoCapture(0)

def dlonie_landmarks(pokaz_id=0):
    '''Funkcja pobiera klatki obrazu z kamery, wykrywa na niej dłonie i zwraca pozycje punktów orientacyjnych'''

    # Wykrywa dłonie oraz rysuje na niej punkty 
    mp_dlonie = mp.solutions.hands
    mp_rysuj = mp.solutions.drawing_utils

    # Ustawia maksymalną ilość wykrywanych dłoi oraz czułość ich detekcji i śledzenia
    dlonie = mp_dlonie.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # Ustawia okienko podglądu (bez tej linijki kamera nie chciała pokazywać obrazu)
    cv2.namedWindow("Podgląd", cv2.WINDOW_NORMAL)


    # Analizuje obiekty znajdujące się w obrazie kamery
    while kamera.isOpened():
        # Sprawdza czy kamera przesyła klatki obrazu
        ret, klatka = kamera.read()
        if not ret:
            print("❌ Brak klatki")
            break

        # Odbija obraz w poziomie, tak żeby gdy prawa ręka zostanie podniesiona, na ekranie również prawa ręka jest podniesiona
        klatka = cv2.flip(klatka, 1)

        # Konwertuje obraz z BGR na RGB, bez którego MediaPipe nie wykrywa dłoni
        rgb = cv2.cvtColor(klatka, cv2.COLOR_BGR2RGB)

        # Wysyła obraz do MediaPipe, który analizuje klatkę, szuka dłoni oraz zwraca jej landmarki
        results = dlonie.process(rgb)

        # Sprawdza czy na kamerze jest wykryta dłoń lub dłonie
        if results.multi_hand_landmarks:
            # Jeżeli wykryta została dłoń
            for reka in results.multi_hand_landmarks:
                # Rysuje punkty na ręce 
                mp_rysuj.draw_landmarks(klatka, reka, mp_dlonie.HAND_CONNECTIONS)
                
                # Importuje klasę "Gesty" z moduły gesty.py
                gest = gesty.Gesty(reka, klatka)

                # Definiuje listę gestów które przy pomocy lambdy się uruchamiają
                lista_gestow = [
                    lambda: gest.kciuk_gora_dol(rysuj=False
                    ),
                    lambda: gest.otwarta_dlon(rysuj=False),
                    lambda: gest.znak_ok(rysuj=False),
                    lambda: gest.pokoj(rysuj=True)
                ]

                # Definiuje wynik, czyli dany gest
                wynik = None

                # Pętla która wywołuje daną funkcję z listy
                for funkcja in lista_gestow:
                    wynik = funkcja()
                    if wynik:
                        break

                # Jeżeli program rozpozna jakiś gest, to wyprintuje informacje o nim, w przeciwnym wypadku jeżeli gestu nie rozpozna da info
                if wynik:
                    print(wynik)
                else:
                    print("✅ Dłoń wykryta")
                
                # Pętla odpowiadająca za pokazanie numeru ID każdej części palca jeżeli pokaz_id == 1
                if pokaz_id == 1:
                    for i, lm in enumerate(reka.landmark):
                        h, w, _ = klatka.shape
                        x, y = int(lm.x * w), int(lm.y * h)

                        cv2.putText(
                            klatka,
                            str(i),
                            (x,y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 50, 255),
                            1
                        )            

        else:
            # W przeciwnym wypadku program informuje o braku dłoni
            print("❌ Brak dłoni")

        # Pokazuje obraz z kamery
        cv2.imshow("Podgląd", klatka)
        # Ustawia przycisk "q" jako zakończenie pętli
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Zwalnia kamerę (kończy jej działanie)
    kamera.release()
    # Zamyka okna stworzone przez cv2 (okno z podglądem kamery)
    cv2.destroyAllWindows()


# Wywołuje funkcję 
dlonie_landmarks(pokaz_id=1)