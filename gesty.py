# Ściąga z ID punktów palców 
#-------------------------
# Palec:     ID punktów
# ------------------------
# Kciuk     → 1, 2, 3, 4
# Wskazujący→ 5, 6, 7, 8
# Środkowy  → 9,10,11,12
# Serdeczny →13,14,15,16
# Mały      →17,18,19,20


import math
import cv2
import mediapipe as mp

class Gesty:
    '''Klasa przechowująca wszystkie funkcje definiujące gesty dłoni'''
    def __init__(self, reka, klatka):
        self.reka = reka
        self.klatka = klatka 


    def otwarta_dlon(self, rysuj=False):
        '''Funkcja charakteryzuje wygląd ręki z otwartymi palcami
        :reka: zmienna definiującą rękę
        :rysuj: True - rysuje połączenia między charakterystycznymi punktami
        :return: napis jeżeli kciuk faktycznie będzie w górze'''
        knykcie_ids = [1, 5, 9, 13, 17]  

        # Kciuk: czubek i staw
        kciuk_tip = self.reka.landmark[4]
        kciuk_pip = self.reka.landmark[2]
        pozostale_otwarte = True
        otwarta_dlon = True
        obrocona_reka = True
        
        for knykiec_id in knykcie_ids:
            knykcie = self.reka.landmark[knykiec_id]
            czubek = self.reka.landmark[knykiec_id + 3]
            czubek_kciuk = self.reka.landmark[4]
            czubek_maly = self.reka.landmark[20]

            knykiec_px, knykiec_py = konwert_znp(knykcie, self.klatka)
            czub_px, czub_py = konwert_znp(czubek, self.klatka)

            dystans_palce = oblicz_odleglosc(knykcie, czubek, wymiary=2)

            czub_kciuk_px, czub_kciuk_py = konwert_znp(czubek_kciuk, self.klatka)
            czub_maly_px, czub_maly_py = konwert_znp(czubek_maly, self.klatka)

            dystans_cz_maly_kciuk = oblicz_odleglosc(czubek_kciuk, czubek_maly, wymiary=2)

            if rysuj == True:
                rysowanie_odleglosci(self.klatka, dystans_palce, knykiec_px, knykiec_py, czub_px, czub_py)
                rysowanie_odleglosci(self.klatka, dystans_cz_maly_kciuk, czub_kciuk_px, czub_kciuk_py, czub_maly_px, czub_maly_py)


            if knykcie.y < czubek.y:  # palec wyprostowany
                pozostale_otwarte = False
                break    

            elif dystans_palce < 0.12:
                otwarta_dlon = False

            elif dystans_cz_maly_kciuk < 0.30:
                obrocona_reka = False

        if pozostale_otwarte and otwarta_dlon and obrocona_reka:
            return "🖐️ Witaj"
        return None

    def kciuk_gora_dol(self, rysuj=False):
        '''Funkcja charakteryzuje wygląd ręki z kciukiem podniesionym do góry lub w dół
        :reka: zmienna definiującą rękę
        :rysuj: True - rysuje połączenia między charakterystycznymi punktami
        :return: napis jeżeli kciuk faktycznie będzie w górze'''
        czubki_ids = [8, 12, 16, 20]  # palce inne niż kciuk

        # Kciuk: czubek i staw
        kciuk_tip = self.reka.landmark[4]
        kciuk_pip = self.reka.landmark[2]
        staw_o_jeden_dalej_wskazujacego = self.reka.landmark[6]
        kciuk_od_staw_odleglosc = False
        reka_dol = False
        reka_gora = False
        zamkniete_palce = True

        lm4 = self.reka.landmark[4]
        lm6 = self.reka.landmark[6]

        # Konwertowanie znormalizowanych wspólrzędnych na pikselowe
        x4_px, y4_px = konwert_znp(lm4, self.klatka)
        x6_px, y6_px = konwert_znp(lm6, self.klatka)

        # Obliczanie odleglosci
        dystans_kciuk = oblicz_odleglosc(lm4, lm6, wymiary=2)

        if rysuj == True:
            rysowanie_odleglosci(self.klatka, dystans_kciuk, x4_px, y4_px, x6_px, y6_px)

        for czubek_id in czubki_ids:
            czubek = self.reka.landmark[czubek_id]
            staw = self.reka.landmark[czubek_id - 3]
            
            czub_px, czub_py = konwert_znp(czubek, self.klatka)
            staw_px, staw_py = konwert_znp(staw, self.klatka)

            dystans_palce = oblicz_odleglosc(czubek, staw, wymiary=2)
            
            if rysuj == True:
                rysowanie_odleglosci(self.klatka, dystans_palce, czub_px, czub_py, staw_px, staw_py)

            if dystans_kciuk > 0.15:
                kciuk_od_staw_odleglosc = True

            if dystans_palce > 0.14:
                zamkniete_palce = False

            if kciuk_tip.y > lm6.y:
                reka_dol = True
            elif kciuk_tip.y < lm6.y:
                reka_gora = True

        if kciuk_od_staw_odleglosc and reka_gora and zamkniete_palce:
            return "👍 Dobrze"
        elif kciuk_od_staw_odleglosc and reka_dol and zamkniete_palce:
            return "👎 Źle"
        return None

    def znak_ok(self, rysuj=False):
        '''Funkcja charakteryzuje wygląd ręki wykonującej gest "ok":
        :reka: zmienna definiującą rękę
        :rysuj: True - rysuje połączenia między charakterystycznymi punktami
        :return: napis jeżeli kciuk faktycznie będzie w górze'''
        czubki_ids = [12, 16, 20] # uwzględnienie tylko palca środkowego, serdecznego i małego

        kciuk_tip = self.reka.landmark[4]
        wskazujacy_tip = self.reka.landmark[8]
        kc_wsk_pol = True
        wyprostowane_palce = True

        kciuk_px, kciuk_py = konwert_znp(kciuk_tip, self.klatka)
        wskaz_px, wskaz_py = konwert_znp(wskazujacy_tip, self.klatka)

        dystans_kciuk_wskazujacy = oblicz_odleglosc(kciuk_tip, wskazujacy_tip, wymiary=2)

        if rysuj == True:
            rysowanie_odleglosci(self.klatka, dystans_kciuk_wskazujacy, kciuk_px, kciuk_py, wskaz_px, wskaz_py)

        for czubek_id in czubki_ids:
            czubek = self.reka.landmark[czubek_id]
            staw = self.reka.landmark[czubek_id - 3]

            czub_px, czub_py = konwert_znp(czubek, self.klatka)
            staw_px, staw_py = konwert_znp(staw, self.klatka)

            dystans_palce = oblicz_odleglosc(czubek, staw, wymiary=2)
            
            if rysuj == True:
                rysowanie_odleglosci(self.klatka, dystans_palce, czub_px, czub_py, staw_px, staw_py)

            if dystans_kciuk_wskazujacy > 0.06:
                kc_wsk_pol = False
                break

            elif dystans_palce < 0.14:
                wyprostowane_palce = False

        if kc_wsk_pol and wyprostowane_palce:
            return "👌 Ok"
        return None


def konwert_znp(punkty, klatka):
    '''Funkcja konwertuje znormalizowane współrzędne na pikselowe na klatce
    :punkty: Miejsce z którego bierze punkty
    :klatka: Klatka z której ma brać współrzędne
    :return: px, py = pikselowe współrzędne punktów'''
    
    h_klatki, w_klatki, _ = klatka.shape
    px, py = int(punkty.x * w_klatki), int(punkty.y * h_klatki)
    return px, py
    

def oblicz_odleglosc(punkt1, punkt2, wymiary=2):
    '''Funkcja liczy odległość w 2 lub 3 wymiarach używając wzoru euklidesa:
    :punkt1: Pierwszy landmark
    :punkt2: Drugi landmark
    :wymiary: ilość wymiarów (2 dla 2D, 3 dla 3D)
    :return: Obliczona odległość    
    '''

    dx = punkt1.x - punkt2.x
    dy = punkt1.y - punkt2.y

    if wymiary == 3:
        dz = punkt1.z - punkt2.z
        odleglosc = math.sqrt(dx**2 + dy**2 + dz**2)
    elif wymiary == 2:
        odleglosc = math.sqrt(dx**2 + dy**2)
    else:
        raise ValueError("Liczba wymiarów musi być 2 lub 3.")

    return odleglosc 

def rysowanie_odleglosci(klatka, dystans, x1, y1, x2, y2):
    '''Funkcja wizualizuje odległość pomiędzy dwoma punktami (x1, y1), (x2, y2):
    :klatka: Klatka na której liczone są współrzędne
    :dystans: Faktyczny dystans
    :x1: Pierwsza współrzędna x
    :y1: Pierwsza współrzędna y
    :x2: Druga współrzędna x
    :y2: Druga współrzędna y
    :Return: Narysowana odległość'''
    try:
        cv2.line(klatka, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Wyświetlenie wartości odległości na linii
        text_x = int((x1 + x2) / 2)
        text_y = int((y1 + y2) / 2) - 10

        rys = cv2.putText(klatka, f"Odleg: {dystans:.2f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 155), 2)
    
    except IndexError:
        print(f"Błąd - nie można uzyskać dostępu do landmarków")
    except Exception as e:
        print(f"Inny błąd podczas wizualizacji odległości: {e}")

    return rys