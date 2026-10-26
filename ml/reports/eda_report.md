# Raport Eksploracyjnej Analizy Danych (EDA)
System automatycznej kontroli jakości odlewów (wirniki pomp zatapialnych)

## 1. Statystyki ogólne i balans klas

- **Łączna liczba obrazów:** 6000
- **Klasa `ok_front` (Poprawny (OK)):** 3000 obrazów (50.00%)
- **Klasa `def_front` (Wadliwy (DEF)):** 3000 obrazów (50.00%)
- **Stosunek klas (def/ok):** 1.00:1 (idealny balans 50/50)

## 2. Integralność plików i format

- **Uszkodzone pliki:** 0
- **Wymiary obrazów (szerokość, wysokość):** [(300, 300)]
- **Tryb kolorów (PIL mode):** ['RGB']
- **Zakres wartości pikseli:** min=0, max=255, średnia=144.55

## 3. Analiza duplikatów i ryzyka wycieku danych

- **Dokładne duplikaty plików (identyczny MD5):** 0 grup(y)
- **Duplikaty percepcyjne (bliskie cechy wizualne dHash):** 270 grup(y)
  *Brak fizycznych duplikatów plików w zbiorze danych.*

