# CluVRP
Celem projektu była implementacja algorytmów rozwiązujących problem Clustered VRP. W tym wariancie Vehicle Routing Problem klienci są podzieleni na klastry i konieczne jest odwiedzenie wszystkich klientów znajdujących się w klastrze przed obsłużeniem innych klientów. W celu rozwiązania problemu zaimplementowano cztery algorytmy: algorytm genetyczny, algorytm cząsteczkowy, algorytm najbliższego wierzchołka i dwupoziomowy algorytm VNS.
## Sposób uruchomienia
W celu uruchomienia poszczególnych algorytmów należy użyć następujących poleceń uzupełnionych o numer zestawu danych jako ostatni parametr. Istnieją zestawy o numerach 1 i 2. Przy podaniu innego numeru zostanie użyty zestaw domyślny.
### Algorytm dwupoziomowy VNS
*py two_level_vns.py* oraz podanie kolejno pojemności pojazdu, parametru **cluVNSProb** oraz parametru **PertRate**
### Algorytm cząsteczkowy
*py hybrid_pso.py* oraz podanie kolejno parametrów **w**, **c1** i **c2** oraz **r1** i **r2**
### Algorytm genetyczny
*py genetic.py* oraz podanie kolejno liczby epok, rozmiaru populacji i szansy na mutację
### Algorytm najbliższego wierzchołka
*py project.py*
