"""Unit tests for the `filesystem` task transformer."""

from __future__ import annotations

import unittest

from filesystem.solve_filesystem import (
    NotesBundle,
    build_batch_actions,
    build_marketplace_data,
)


ANNOUNCEMENTS = """--- TABLICA OGLOSZEN ---

Kto bedzie przejezdzal przez Opalino, niech podrzuci 45 chlebow, 120 butelek wody i 6 mlotkow, temat pilny od rana.

Do Domatowa trzeba dorzucic na transport 60 makaronu, 150 butelek wody i 8 lopat, bo zapasy siadly.

Brudzewo: ryz 55 workow + 140 butelek wody + 5 wiertarek, najlepiej jeszcze dzisiaj.

Po rozmowie z kierowca wyszlo, ze w Darzlubiu schodzi towar szybko, potrzeba 25 porcji wolowiny, 130 butelek wody i 7 kilofow.

Notka z rana: Celbowo pyta o 40 porcji kurczaka, 125 butelek wody i 6 mlotkow.

Uwaga na kurs pod Mechowo, tam duzy zrzut: ziemniaki 100 kg, kapusta 70, marchew 65 kg, woda 165 butelek, lopaty 9.

Ktos pytal o Puck, na ten moment potrzebuja 50 chlebow, 45 workow ryzu, 175 butelek wody i 7 wiertarek.

Na koniec Karlinkowo: do uzupelnienia 52 makaronu, 22 porcje wolowiny, 95 kg ziemniakow, 155 butelek wody i 6 kilofow.
"""

CONVERSATIONS = """= Notatki przygotowane przez Natana Ramsa z Domatowa =

- rano obdzwanialem wszystkich po kolei. U mnie w Domatowie sprawa prosta, ja to spinam, tylko trzeba dopchac makaron, wode i lopaty zanim znowu zaczna marudzic z placu. Natan Rams, jak zwykle, wszystko na swojej glowie.

- z Opalina dzwonila dzisiaj Iga Kapecka, mowila ze chleb jeszcze jest liczony na styk, ale tak naprawde trzeba im dowiezc chleb, wode i mlotki. Ma dac znac jutro, czy uda sie to zrzucic jednym autem.

- udalo mi sie zalatwic troche wody dla Brudzewa. Kisiel ma do mnie dzwonic jeszcze w sprawie ryzu i wiertarek pod koniec tygodnia, bo bez tego nie domkna brakow.

- Rafal oddzwonil wieczorem. Woda dla Brudzewa bedzie szybciej, ale z ryzem i wiertarkami moze byc opoznienie, bo sciagaja to z dwoch roznych miejsc.

- krotka rozmowa z Darzlubiem: naciskaja na kilofy i wode, a do tego jeszcze wolowina ma byc dowieziona, bo zapasy jedzenia siadly. Marta Frantz brzmiala, jakby juz trzeci dzien nie spala, co w sumie moze byc prawda przy tym, co tam sie dzieje.

- z Celbowa sygnal dosc spokojny, ale braki sa konkretne: jezo skurczaka, woda i mlotki. Oskar Radtke ma przeslac konkretne liczby wieczorem, jak domkna stan.

- Mechowo znowu na linii. Eliza Redmann dzwonila dwa razy. Oba telefony o tym samym: trzeba im ziemniaki, kapuste, marchew, wode i lopaty, bo z samych resztek tygodnia nie pociagna.

- z Darzlubiem trudno robic interesy. Ta Frantz jak sie uprze, to dzwoni bez konca. Jak powie, ze ma byc wolowina, woda i kilofy, to mam jej to wyczarowac spod ziemi. Przynajmniej wode juz sobie jakos wykombinowali.

- z Pucka dzwonil Damian Kroll, glos jak zawsze spokojny, ale temat powazny. Chce domknac chleb, ryz, wode i wiertarki jednym transportem, osobno nie chce mu sie tego rozbijac.

- Karlinkowo odkrecilo sie dopiero po poludniu. Najpierw krotki sygnal od Konkel, potem dluzsza rozmowa. Teraz to Lena pilnuje tam handlu i prosi, zeby ziemniaki, makaron i wolowina jechaly razem, bo nie ma komu drugi raz odbierac. Reszta moze poczekac.
"""

TRANSACTIONS = """Darzlubie -> ryż -> Puck
Puck -> marchew -> Mechowo
Domatowo -> chleb -> Opalino
Opalino -> wołowina -> Darzlubie
Puck -> kilof -> Darzlubie
Karlinkowo -> wiertarka -> Puck
Celbowo -> chleb -> Opalino
Brudzewo -> mąka -> Karlinkowo
Karlinkowo -> młotek -> Opalino
Opalino -> makaron -> Domatowo
Celbowo -> kapusta -> Mechowo
Domatowo -> ziemniaki -> Mechowo
Opalino -> ryż -> Brudzewo
Mechowo -> kilof -> Karlinkowo
Brudzewo -> chleb -> Puck
Darzlubie -> ziemniaki -> Karlinkowo
Darzlubie -> kurczak -> Celbowo
Karlinkowo -> ryż -> Brudzewo
Brudzewo -> łopata -> Domatowo
Puck -> łopata -> Domatowo
Mechowo -> mąka -> Domatowo
Mechowo -> młotek -> Celbowo
Celbowo -> kilof -> Darzlubie
Domatowo -> wiertarka -> Brudzewo
"""


class SolveFilesystemTests(unittest.TestCase):
    def test_build_marketplace_data_extracts_expected_structure(self) -> None:
        data = build_marketplace_data(
            NotesBundle(
                announcements=ANNOUNCEMENTS,
                conversations=CONVERSATIONS,
                transactions=TRANSACTIONS,
            )
        )

        self.assertEqual(
            data.city_needs["mechowo"],
            {
                "ziemniak": 100,
                "kapusta": 70,
                "marchew": 65,
                "woda": 165,
                "lopata": 9,
            },
        )
        self.assertEqual(data.city_managers["brudzewo"], "Rafal Kisiel")
        self.assertEqual(data.city_managers["karlinkowo"], "Lena Konkel")
        self.assertEqual(
            data.goods_sources["chleb"],
            ["brudzewo", "celbowo", "domatowo"],
        )
        self.assertEqual(
            data.goods_sources["maka"],
            ["brudzewo", "mechowo"],
        )

    def test_build_batch_actions_uses_expected_paths_and_links(self) -> None:
        data = build_marketplace_data(
            NotesBundle(
                announcements=ANNOUNCEMENTS,
                conversations=CONVERSATIONS,
                transactions=TRANSACTIONS,
            )
        )

        actions = build_batch_actions(data)

        self.assertEqual(actions[0], {"action": "reset"})
        self.assertIn({"action": "createDirectory", "path": "/miasta"}, actions)
        self.assertIn(
            {
                "action": "createFile",
                "path": "/osoby/rafal_kisiel",
                "content": "Rafal Kisiel\n[Brudzewo](/miasta/brudzewo)",
            },
            actions,
        )
        self.assertIn(
            {
                "action": "createFile",
                "path": "/towary/chleb",
                "content": "- [Brudzewo](/miasta/brudzewo)\n- [Celbowo](/miasta/celbowo)\n- [Domatowo](/miasta/domatowo)",
            },
            actions,
        )


if __name__ == "__main__":
    unittest.main()
