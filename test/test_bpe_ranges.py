import xlingqg_server.translation


def test_bpe():
    a = xlingqg_server.translation.bpe_ranges(['This', 'is', 'a', 'sub@@'
        , 'wo@@', 'rd', 'te@@', 'st@@', 'case', '.'])

    assert a == [(0,0),(1,1),(2,2),(3,5),(6,8),(9,9)]