# rmtt_field — софт для натурних дослідів із RoboMaster TT

Керування дронами RoboMaster TT (Tello Talent, SDK 3.0), запис відео й телеметрії, місії з переднім ToF як захистом від зіткнень, експорт записів облету в журнал АС Р1 (`ac1_evacmap`).

**На справжніх дронах ще не перевірено.** Усе перевірено на емуляторі (`test_field.py`, 34 перевірки). Порядок дослідів — `PROTOCOL.md`; бриф для Claude виконавця — `CLAUDE.md`.

## Модулі

| файл | що робить |
|---|---|
| `tello_link.py` | три сокети UDP (команди 8889, стан 8890, відео 11111), розбирання за IP дрона; відео пишеться сирим H.264 з часом кожного пакета, без декодування в польоті |
| `drone.py` | один дрон: команди з підтвердженням, запити, `EXT tof?`, світлодіоди й номер на матриці, рух частинами з перевіркою ToF, відновлення після втраченого підтвердження за телеметрією |
| `safety.py` | спільне переривання; ПРОБІЛ — посадка всіх, E E — зупинка двигунів, Q — перервати місію (Windows); Ctrl+C — посадка на будь-якій ОС |
| `missions.py` | місії `bench`, `hover`, `line`, `survey`; передпольотна перевірка; монітор заряду й зв'язку |
| `fleet.py` | `fleet.json`, пошук дронів у мережі роутера, перехід у режим роутера |
| `postprocess.py` | час кожного кадру за розбором потоку H.264, `video.mp4`, `frames.csv`, експорт у журнал АС Р1 (формат відтворено, код АС не потрібен) |
| `make_mission.py` | місія облету сцени з перевіркою відстаней до стін і предметів (`ground_truth/<сцена>_objects.csv`) |
| `calibrate_camera.py` | калібрування камери за записом шахівниці (`checkerboard_A4.pdf`) |
| `pack_results.py` | архів результатів + `results_summary.csv` |
| `export_field.py` | усі успішні обльоти розпакованого набору -> журнали АС Р1 (калібрування за SN, положення з рисканням) |
| `sim_rmtt.py` | емулятор RMTT: команди SDK 3.0, стан 10 Гц, відео 30 к/с, приміщення з ящиками, втрати пакетів |
| `app.py` | командний рядок |
| `test_field.py` | автоматичні сценарії на емуляторі, пише `test_log.txt` |

## Команди

```
pip install -r requirements.txt
python test_field.py                                        # без дронів, близько 3 хв
python app.py bench --ap --label 3                          # стенд дрона з наліпкою 3 (пряме підключення)
python app.py bench --ap --label 3 --seconds 45 --name calib
python calibrate_camera.py runs/<тека> --out calib/calib_d3.json
python app.py run missions/hover.json --ap --label 3 --decode
python make_mission.py --scene S2 --room 7.0 5.8 --origin 1.0 1.0 --waypoints "1,1 1,5.1 5.2,5.1 5.2,1" --gt ground_truth/S2_objects.csv
python app.py run missions/survey_S2.json --ap --label 3 --decode
python app.py decode runs/<тека>
python app.py panic --ap                                    # з іншого вікна
python pack_results.py
```

Репетиція з емулятором: в одному вікні `python app.py sim --n 1`, в іншому ті самі команди з `--ap-ip 127.0.0.11` перед назвою команди, наприклад `python app.py --ap-ip 127.0.0.11 run missions/hover.json --ap --label 1`.

Флот через роутер (не потрібен для протоколу): `setup-router`, `scan`, `--drones 1,2`, `--assign 1=a.json 2=b.json`.

## Тека запису `runs/<дата_час>_<місія>_d<номер>/`

`mission.json`, `meta.json` (дрон, серійний номер, результат, причина переривання), `marks.json` (зліт, висота, точки, моменти зйомки), `commands.csv` (кожна команда з часом і відповіддю, примітки про відновлення), `state.csv` (стан 10 Гц), `ext_tof.csv`, `video.h264` + `video_index.csv`. Після `decode`: `video.mp4`, `frames.csv`, `decode_report.json`, `first_frame.png`.

## Системи відліку

Кімната: x уздовж обраної стіни, y ліворуч, кути проти годинникової стрілки (як в АС Р1). Tello: рискання за годинниковою стрілкою додатне. `yaw_АС = heading_deg + yaw_sign · (yaw_Tello − yaw_Tello на старті)`, типово `yaw_sign = −1`; знак визначається в польоті (hover: «cw 90») і за поворотами кожного облету під час експорту. Положення в точках — числення шляху за виконаними командами; еталон — позначки на підлозі й рулетка.

## Що не перевірено на апараті

Формати відповідей `EXT tof?`, `EXT led`, `EXT mled`; знак рискання; роздільність потоку; частота стану. Усе це показує `bench`.
