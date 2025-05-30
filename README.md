Jest to skrypt który pozwala za pomocą kamery internetowej rozpoznać pokazane gesty (na chwilę obecną działają tylko cztery: 👍, 👎, 🖐️, 👌).

Skrypt należy uruchomić przez moduł kamera.py. Korzysta on z bibliotek cv2 oraz mediapipe.

Aby skrypt działał poprawnie, należy korzystać z pythona w wersji 3.10, ponieważ biblioteka mediapipe, na której bazowany jest cały kod, nie wspiera najnowszej wersji pythona 3.13.

Na chwilę obecną program może momentami źle definiować gest "👌" i rozumieć go jako "🖐️". W takim wypadku należy przybliżyć lub oddalić rękę. Jest to bug nad którym pracuje.

W planach na przyszłość jest zaimplementowanie większej ilości gestów, poprawienie dokładności ich wykrywania (implementacja osi Z w programie) oraz dodania funkcjonalności do nich (do dopracowania).
