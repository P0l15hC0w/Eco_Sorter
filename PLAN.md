## Plan aplikacji, pomysly, i tym podobne.
### dzialanie aplikacji:
- przewidywanie wartosci i zapisywanie danych w zmiennej
- normalizacja danych, skalowanie i debounce w formie werdyktu, mapowanie klasy
- przekazywanie danych do reki robotycznej w formie pakietu
- plytka sterujaca dlonia odbiera pakiety i zapisuje w kolejce wykonywania
- pakiet sklada sie z: ({pozycja xy odpadu}, {indeks klasy odpadu}, {stop w przypadku reki(boolean)})

### klasy modelu wykrywania dloni:
0. dlon lub reka
1. neutral
2. rzecz

### klasy modelu odpadow:
0. neutralna (zmieszane)
1. plastiki i metale
2. papier
3. szklo
4. odpady bio

### klasy modeli lokalizacji:
#### model x:
0. K1 (kolumna 1)
1. K2 (kolumna 2)
2. K3 (kolumna 3)
3. K4 (kolumna 4)
4. neutralny
#### model y:
0. R1 (rzad 1)
1. R2 (rzad 2)
2. R3 (rzad 3)
3. R4 (rzad 4)
4. neutralny

### kolejnosc dzialania modeli ai:
1. model dloni 
    - sprawdza czy dlon pojawia sie w miejscu pracy robota
    - jezeli dlon pojawia sie, robot zatrzymuje prace
    - jesli dlon znika robot przechodzi do nastepnego modelu ai
2. model odpadow
    - przewiduje rodzaj smieci
    - gdy odpowiednio przewidzi klase, mapuje kontener i przechodzi do nastepnego modelu
3. modele lokalizacji (matryca xy)
    - przewiduje lokalizacje odpadu na powierzchni pracy
    - uzywa kolumn i rzedow jako klas