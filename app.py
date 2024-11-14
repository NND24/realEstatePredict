from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pyodbc

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def map_district_level(DistrictId):
    if DistrictId in [1]:
        return 40
    elif DistrictId in [3, 5]:
        return 25
    elif DistrictId in [10]:
        return 22
    elif DistrictId in [17]:
        return 17
    elif DistrictId in [11,15,18]:
        return 14
    elif DistrictId in [12]:
        return 2
    elif DistrictId in [6, 16, 19]:
        return 10
    elif DistrictId in [7]:
        return 9
    elif DistrictId in [8, 14]:
        return 8
    elif DistrictId in [13]:
        return 7
    elif DistrictId in [24]:
        return 6
    elif DistrictId in [4, 9, 12]:
        return 5
    elif DistrictId in [23]:
        return 3
    elif DistrictId in [20,21]:
        return 2
    elif DistrictId in [22]:
        return 1
    else:
        return 0
    
def map_street_level(StreetId):
    if StreetId in [13]:
        # 1.5
        return 150
    elif StreetId in [6]:
        # 1.4
        return 140
    elif StreetId in [18]:
        # 1.3
        return 130
    elif StreetId in []:
        # 1.2
        return 120
    elif StreetId in [12,24]:
        # 1.1
        return 110
    elif StreetId in [23]:
        # 1.0
        return 100
    elif StreetId in [30,98]:
        # 0.9
        return 90
    elif StreetId in [19]:
        # 0.8
        return 80
    elif StreetId in [1,16,32,51,82,96]:
        # 0.7
        return 70
    elif StreetId in [4,5,11,15,22,27,28,43,84,87,90,93,291,294,1083,2071]:
        # 0.6
        return 60
    elif StreetId in [3,10,17,25,33,34,36,42,54,67,85,92,302,264,281,1054,1169]:
        # 0.5
        return 50
    elif StreetId in [26,48,52,58,59,61,66,67,76,81,83,88,94,95,104,106,289,292,293,301,259,284,312,396,398,400,415,467,1009,1022,2032,2100,2152]:
        # 0.4
        return 40
    elif StreetId in [20,21,31,40,47,53,55,57,60,62,63,64,65,73,74,75,77,80,89,97,102,112,136,288,290,295,297,298,299,300,261,262,266,272,273,278,282,283,286,306,308,322,333,381,382,385,386,390,391,392,393,395,399,401,402,405,410,411,414,417,428,429,431,435,436,441,442,443,444,450,453,454,460,461,463,471,478,488,1007,1021,1032,1043,1057,1062,1065,1074,1082,1092,1097,1148,1150,1200,1273,1767,2022,2065,2108,2116,240]:
        # 0.3
        return 30
    elif StreetId in [8,35,37,38,45,46,49,50,56,69,70,71,72,78,79,91,99,101,103,107,108,109,110,111,113,114,115,116,122,124,130,131,132,134,135,140,141,146,153,154,186,199,201,202,203,204,205,209,215,220,221,222,227,229,257,260,263,265,267,268,269,270,271,274,275,276,277,279,280,304,305,307,309,316,318,320,321,323,325,359,389,394,407,416,420,422,423,424,426,427,430,432,433,439,440,446,447,449,451,452,456,462,464,465,474,476,482,487,509,526,527,528,529,530,531,533,544,545,549,558,559,560,564,577,604,605,618,619,620,630,634,637,638,639,640,641,649,657,658,661,668,672,675,677,692,693,694,695,710,727,803,809,1010,1011,1012,1015,1016,1018,1020,1023,1024,1026,1028,1029,1030,1034,1036,1037,1038,1039,1040,1041,1044,1045,1046,1049,1053,1056,1058,1060,1061,1063,1066,1067,1068,1069,1072,1073,1076,1077,1078,1079,1080,1081,1084,1086,1087,1088,1089,1091,1093,1094,1095,1101,1102,1104,1105,1106,1107,1108,1113,1115,1116,1117,1118,1120,1121,1124,1130,1132,1146,1147,1159,1170,1179,1180,1182,1183,1215,1218,1221,1222,1467,1769,1831,1850,1851,1852,1853,1854,1855,1856,1861,1863,2020,2021,2025,2028,2033,2034,2035,2051,2055,2069,2070,2074,2076,2086,2102,2104,2112,2114,2115,2129,2132,2136,2137,2141,2142,2143,2144,2147,2148,2149,2150,2151,2154,2156,2163,2164,2165,2169,2170,2172,2175,2177,2178,2182,2206,2225,2227,2234,2241,2242,2243,2244,2248,2249,2250,2254,2255,2277,2278,2281,2350,2354,2423,233,237,238,242,245]:
        # 0.2
        return 20
    elif StreetId in [255,248,249,247,246,244,243,14,39,41,105,121,123,125,126,127,128,129,137,138,139,143,144,147,149,150.151,155,156,167,173,179,191,194,200,206,207,208,210,211,212,216,217,218,224,226,230,231,258,285,303,311,313,315,317,319,324,336,352,378,383,384,387,408,409,418,448,457,466,470,479,480,481,483,489,492,494,499,500,501,502,508,510,512,513,518,519,520,522,523,525,532,534,535,536,537,539,541,542,543,546,547,553,554,555,556,557,561,562,566,568,569,570,571,572,573,574,575,576,578,579,580,583,584,585,586,590,595,598,601,603,608,617,621,622,627,633,635,636,642,643,644,645,647,648,650,651,652,653,654,655,660,665,670,673,676,681,690,691,696,697,698,699,700,701,702,703,704,705,706,707,708,709,711,715,725,726,728,731,733,734,735,737,740,742,743,744,746,748,753,754,756,759,763,766,767,768,769,772,775,776,777,778,779,780,781,783,786,790,791,793,794,795,799,838,843,844,845,846,848,853,857,858,860,861,882,926,927,929,930,931,932,945,982,1008,1013,1019,1025,1031,1033,1047,1048,1050,1052,1055,1059,1064,1071,1075,1085,1090,1096,1098,1100,1109,1110,1111,1112,1114,1119,1122,1123,1126,1127,1128,1129,1131,1134,1136,1139,1144,1154,1155,1156,1158,1161,1162,1165,1171,1173,1175,1176,1184,1186,1187,1190,1192,1193,1197,1198,1199,1201,1202,1204,1205,1206,1208,1209,1210,1211,1212,1214,1217,1220,1223,1224,1225,1226,1241,1289,1315,1316,1377,1378,1379,1380,1383,1408,1418,1421,1435,1448,1454,1475,1506,1511,1522,1526,1528,1531,1532,1535,1542,1560,1561,1562,1564,1569,1573,1587,1604,1622,1676,1677,1678,1679,1681,1682,1683,1684,1685,1686,1687,1688,1689,1690,1691,1692,1693,1694,1695,1696,1697,1698,1699,1700,1701,1702,1703,1704,1705,1706,1707,1708,1709,1710,1711,1712,1717,1719,1721,1723,1725,1746,1747,1748,1749,1750,1751,1752,1753,1755,1756,1757,1759,1761,1765,1766,1768,1770,1771,1772,1773,1774,1776,1777,1778,1779,1780,1781,1783,1784,1785,1786,1787,1789,1790,1791,1792,1793,1794,1795,1796,1797,1798,1799,1800,1802,1804,1805,1807,1808,1809,1810,1812,1813,1814,1818,1819,1820,1821,1822,1828,1830,1832,1833,1834,1838,1839,1841,1842,1843,1844,1846,1847,1848,1849,1857,1858,1859,1860,1862,1864,1865,1867,1871,1872,1873,1877,1879,1880,1881,1885,1887,1888,1889,1890,1893,1895,1898,1899,1900,1903,1906,1907,1908,1909,1912,1913,1916,1917,1919,1920,1922,1923,1925,1926,1927,1928,1929,1930,1931,1932,1941,1942,1943,1953,1954,1955,1956,1958,1959,1961,1962,1966,1972,1973,1976,1978,1987,1989,2002,2007,2009,2011,2013,2014,2015,2017,2018,2019,2023,2026,2027,2029,2030,2031,2038,2039,2040,2041,2042,2044,2045,2046,2047,2048,2049,2050,2052,2053,2054,2059,2060,2061,2063,2066,2067,2068,2072,2075,2078,2080,2081,2082,2083,2084,2085,2087,2088,2089,2090,2091,2092,2093,2094,2095,2097,2098,2099,2103,2105,2106,2109,2110,2111,2113,2117,2118,2121,2127,2128,2131,2133,2134,2135,2138,2139,2140,2145,2146,2153,2155,2158,2159,2161,2162,2166,2167,2168,2171,2173,2174,2176,2183,2184,2185,2186,2187,2188,2189,2190,2191,2192,2193,2194,2196,2197,2198,2199,2200,2201,2202,2203,2205,2207,2208,2210,2211,2212,2213,2214,2215,2216,2220,2224,2226,2228,2229,2230,2231,2232,2233,2236,2238,2239,2240,2245,2246,2247,2251,2252,2253,2256,2257,2258,2259,2260,2261,2262,2263,2264,2265,2268,2269,2270,2271,2272,2273,2274,2275,2276,2279,2280,2282,2283,2284,2285,2286,2287,2289,2290,2291,2294,2297,2300,2306,2309,2311,2312,2316,2317,2319,2321,2322,2326,2327,2330,2331,2332,2333,2334,2335,2336,2337,2338,2339,2340,2341,2342,2343,2344,2345,2347,2349,2353,2356,2358,2360,2361,2362,2363,2365,2366,2367,2368,2369,2371,2374,2377,2378,2380,2381,2383,2384,2385,2389,2390,2393,2397,2398,2400,2402,2404,2405,2406,2407,2409,2411,2412,2413,2414,2415,2416,2417,2420,2421,2425,2427,2428,2429,2432,2433,2434,2435,2436,2437,2438,2439,2440,2441,2442,2443,2444,2445,2446,2447,2449,2450,2451,2452,2453,2454,2459,2460,2462,2464,2465,2467,2468,2470,2471,2472,2475,2476,2477,2480,2481,2503,2506,2508,2509,2511,2742,2758,232,234,235,236,239,241,252,253]:
        # 0.1
        return 10
    elif StreetId in [9,158,182,195,334,338,552,588,591,594,610,615,632,678,686,713,729,732,749,761,765,770,789,801,808,823,834,839,852,854,856,884,886,887,947,953,967,1042,1099,1151,1152,1177,1178,1324,1384,1386,1391,1405,1409,1436,1473,1519,1529,1530,1533,1534,1539,1548,1550,1551,1552,1553,1554,1555,1556,1557,1558,1563,1565,1566,1570,1571,1576,1580,1594,1597,1598,1599,1608,1610,1612,1613,1618,1630,1639,1659,1669,1722,1815,1817,1824,1825,1826,1829,1836,1875,1876,1883,1884,1892,1897,1904,1915,1934,1936,1938,1939,1940,1946,1947,1948,1949,1950,1963,1964,1965,1967,1971,1975,1977,1980,1981,1985,1986,1990,1991,1994,1999,2001,2005,2008,2016,2064,2119,2120,2123,2124,2125,2179,2180,2181,2217,2223,2266,2267,2293,2298,2301,2305,2307,2315,2318,2323,2325,2329,2352,2355,2357,2359,2362,2386,2388,2392,2394,2395,2403,2410,2418,2430,2456,2457,2466,2474,2479,2715]:
        # 0.09
        return 9
    elif StreetId in [100,162,164,165,170,214,347,354,370,373,517,587,589,624,712,718,720,721,723,739,757,773,837,840,875,901,920,937,948,954,980,983,987,1125,1191,1253,1283,1284,1323,1330,1374,1392,1407,1413,1423,1428,1434,1437,1443,1445,1447,1453,1456,1462,1465,1490,1537,1543,1546,1549,1559,1567,1572,1575,1579,1581,1589,1590,1591,1593,1602,1605,1606,1607,1609,1611,1615,1616,1620,1623,1624,1634,1636,1638,1640,1652,1653,1654,1657,1661,1663,1667,1671,1714,1715,1753,1764,1823,1835,1837,1845,1866,1869,1886,1891,1894,1914,1921,1937,1944,1951,1957,1968,1970,1974,1979,1982,1983,1984,1988,1992,1997,2004,2058,2077,2096,2122,2126,2235,2292,2302,2320,2328,2346,2348,2351,2376,2382,2391,2396,2399,2401,2422,2426,2469,2473,2478,2482,2739,251]:
        # 0.08
        return 8
    elif StreetId in [44,119,120,133,152,159,163,176,181,198,310,330,337,357,376,404,469,540,550,551,563,565,602,607,646,685,687,722,736,755,758,764,787,847,859,868,878,900,906,917,922,924,933,951,957,958,974,977,979,995,997,1001,1145,1195,1228,1229,1238,1256,1258,1259,1265,1271,1275,1291,1309,1322,1328,1338,1353,1363,1372,1382,1387,1388,1390,1393,1395,1396,1403,1406,1410,1412,1414,1415,1416,1417,1419,1422,1426,1432,1438,1440,1452,1457,1459,1461,1463,1464,1466,1476,1487,1488,1499,1503,1513,1516,1517,1521,1523,1524,1544,1568,1577,1582,1584,1586,1596,1614,1617,1619,1621,1631,1635,1651,1656,1658,1662,1668,1672,1716,1720,1724,1728,1816,1868,1870,1878,1905,1910,1933,1935,1952,1996,2006,2010,2073,2218,2221,2222,2303,2304,2372,2387,2419,2448,2461,2463,2507,2716,2740,2741,2750,250]:
        # 0.07
        return 7
    elif StreetId in [148,157,168,169,172,180,183,184,192,228,351,355,374,375,377,445,516,538,581,592,599,625,629,714,716,724,745,782,785,800,814,824,828,867,870,873,876,877,883,885,889,904,913,915,923,938,944,949,956,964,970,981,984,985,991,994,996,1070,1163,1167,1168,1181,1231,1233,1234,1237,1242,1249,1251,1254,1262,1263,1266,1267,1276,1277,1278,1279,1280,1285,1286,1293,1301,1302,1306,1307,1308,1311,1312,1317,1318,1319,1327,1331,1335,1339,1340,1351,1357,1360,1367,1368,1381,1385,1398,1400,1401,1402,1411,1424,1429,1431,1433,1439,1441,1442,1444,1455,1460,1470,1472,1479,1480,1482,1518,1527,1538,1540,1574,1578,1588,1592,1595,1603,1625,1626,1629,1633,1637,1642,1645,1647,1660,1664,1665,1666,1670,1673,1718,1727,1745,1762,1801,1902,1960,1969,1993,2000,2003,2062,2195,2209,2219,2295,2299,2308,2324,2458,2657,2669,2701,2711,2736,2738]:
        # 0.06
        return 6
    elif StreetId in [160,166,171,175,190,196,213,327,2754,349,412,421,425,437,455,468,497,498,503,506,507,514,548,659,662,663,666,829,835,851,862,863,865,866,871,874,879,881,891,893,902,903,907,909,912,914,916,921,946,952,955,959,961,963,968,976,988,992,993,1002,1004,1035,1149,1172,1196,1232,1235,1243,1244,1245,1247,1252,1255,1257,1261,1264,1269,1270,1281,1282,1287,1288,1303,1304,1305,1310,1321,1325,1326,1329,1333,1334,1342,1343,1344,1347,1352,1354,1356,1364,1366,1369,1371,1373,1376,1389,1397,1399,1404,1420,1427,1446,1449,1469,1471,1474,1483,1484,1485,1486,1489,1491,1492,1493,1494,1502,1507,1509,1520,1525,1545,1628,1632,1641,1643,1644,1646,1648,1649,1713,1726,1729,1732,1737,1882,1896,1918,1998,2024,2056,2101,2237,2310,2370,2373,2512,2653,2665,2668,2673,2677,2690,2695,2696,2697,2717,2752,2755]:
        # 0.05
        return 5
    elif StreetId in [296,314,326,328,342,369,379,459,477,493,567,593,606,664,671,674,679,680,682,683,684,688,788,802,822,833,869,897,899,905,908,910,928,935,939,942,962,966,969,972,978,986,989,990,999,1000,1005,1194,1213,1219,1227,1236,1239,1246,1250,1260,1268,1272,1292,1314,1320,1337,1341,1345,1348,1349,1355,1359,1361,1362,1370,1375,1394,1450,1451,1477,1498,1500,1501,1508,1512,1515,1536,1547,1585,1600,1627,1650,1674,1674,1730,1735,1739,1740,1742,1803,1945,2079,2424,2431,2455,2504,2510,2564,2569,2656,2658,2661,2666,2674,2675,2710,2714,2720,2721,2737,2756,2757,2759]:
        # 0.04
        return 4
    elif StreetId in [177,188,189,193,329,331,332,339,340,348,356,366,368,388,403,486,490,491,495,521,524,582,613,669,719,747,774,819,831,832,864,872,880,888,890,892,896,911,918,919,925,934,936,941,950,960,965,975,998,1006,1157,1203,1216,1248,1274,1332,1336,1358,1365,1425,1458,1468,1478,1495,1497,1504,1510,1514,1680,1733,1734,1736,1811,2012,2057,2130,2160,2288,2313,2314,2502,2513,2521,2548,2565,2572,2579,2659,2662,2663,2664,2670,2685,2698,2700,2702,2704,2705,2706,2708,2709,2713,2719,2731,2735,2747,2748,2749,2753,256]:
        # 0.03
        return 3
    elif StreetId in [142,145,161,174,178,185,187,197,223,335,341,344,350,353,358,361,362,363,364,365,367,371,372,380,406,419,434,438,458,472,475,484,485,504,505,511,611,656,717,730,738,750,771,792,804,807,811,816,820,821,827,830,841,894,895,940,943,971,1003,1135,1230,1240,1290,1299,1313,1350,1496,1505,1541,1583,1655,1738,1741,1743,1744,1760,1763,1782,1806,1827,1840,1901,1911,1924,1995,2379,2408,2484,2505,2517,2522,2523,2524,2527,2534,2535,2536,2539,2540,2545,2547,2549,2552,2554,2556,2558,2559,2560,2562,2568,2571,2580,2626.2672,2676,2678,2681,2682,2683,2684,2686,2687,2688,2694,2699,2703,2707,2712,2733,2734,2745,2751,2760,254]:
        # 0.02
        return 2
    elif StreetId in [5,117,118,219,225,343,345,346,360,397,413,473,496,515,596,597,600,609,612,614,616,623,626,628,631,667,689,741,751,752,760,762,784,796,797,798,805,806,810,812,813,817,818,825,826,836,842,849,850,855,898,973,1014,1027,1051,1103,1133,1137,1138,1140,1141,1142,1143,1153,1160,1164,1166,1174,1185,1188,1189,1207,1294,1295,1296,1297,1298,1300,1346,1430,1481,1731,1758,1775,1788,2036,2037,2043,2157,2204,2296,2487,2488,2489,2490,2491,2492,2493,2494,2495,2496,2497,2498,2499,2500,2501,2514,2515,2516,2518,2518,2520,2525,2526,2528,2529,2530,2531,2532,2533,2537,2538,2541,2542,2542,2544,2546,2550,2551,2553,2555,2557,2561,2563,2570,2573,2574,2575,2576,2577,2578,2585,2586,2589,2590,2591,2592,2593,2594,2595,2596,2597,2598,2600,2603,2604,2605,2606,2607,2608,2609,2611,2613,2614,2615,2616,2619,2620,2621,2622,2624,2625,2627,2628,2629,2630,2631,2632,2634,2635,2635,2637,2638,2639,2643,2644,2645,2646,2450,2654,2655,2660,2667,2671,2679,2689,2691,2692,2718,2723,2724,2725,2726,2727,2728,2729,2730,2732,2743,2744,2746]:
        # 0.01
        return 1
    else:
        return 0

# Cấu hình kết nối SQL Server
def get_db_connection():
    conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                          'SERVER=localhost;'
                          'DATABASE=BatDongSan;'
                          'UID=sa;'
                          'PWD=123456789')
    return conn

# API lấy danh sách dữ liệu từ bảng 'real_estates'
@app.route('/getRealEstates', methods=['POST'])
def get_real_estates():
    # Get query parameters from the request
    data = request.get_json()

    category_id = int(data.get('categoryId')) if data.get('categoryId') is not None else None
    price = int(data.get('price')) if data.get('price') is not None else None
    district_id = int(data.get('districtId')) if data.get('districtId') is not None else None
    size = float(data.get('size')) if data.get('size') is not None else None
    rooms = int(data.get('rooms')) if data.get('rooms') is not None else None
    toilets = int(data.get('toilets')) if data.get('toilets') is not None else None
    floors = int(data.get('floors')) if data.get('floors') is not None else None
    estate_type = data.get('type') if data.get('type') is not None else None
    furnishing_sell = data.get('furnishingSell') if data.get('furnishingSell') is not None else None
    urgent = data.get('urgent') if data.get('urgent') is not None else None
    characteristics = data.get('characteristics') if data.get('characteristics') is not None else None

    # Connect to database
    conn = get_db_connection()
    cursor = conn.cursor()

    # Construct base SQL query
    query = 'SELECT * FROM HCMRealEstate WHERE Status = ? AND CategoryId = ? AND DeleteStatus = ?'
    params = ['Đang hiển thị', category_id, 'FALSE']

    # Append filters based on the parameters
    if rooms and rooms > 0:
        query += ' AND Rooms = ?'
        params.append(rooms)

    if toilets and toilets > 0:
        query += ' AND Toilets = ?'
        params.append(toilets)

    if floors and floors > 0:
        query += ' AND Floors = ?'
        params.append(floors)

    if estate_type:
        query += ' AND Type = ?'
        params.append(estate_type)

    if furnishing_sell:
        query += ' AND FurnishingSell = ?'
        params.append(furnishing_sell)

    if characteristics:
        query += ' AND Characteristics = ?'
        params.append(characteristics)

    if urgent:
        query += ' AND Urgent = ?'
        if (urgent == "0"):
            params.append(False)
        else:
            params.append(True)

    if price:
        min_price = price - 1000000000
        max_price = price + 1000000000
        query += ' AND (Price >= ? AND Price <= ?)'
        params.extend([min_price, max_price])

    if size:
        min_size = size - 20
        max_size = size + 20
        query += ' AND (Size >= ? AND Size <= ?)'
        params.extend([min_size, max_size])

    # Execute the query with the parameters
    cursor.execute(query, params)
    real_estates = cursor.fetchall()

    # Close the database connection
    conn.close()

    # Format the results as a list of dictionaries
    real_estates_list = [
        {
            'RealEstateId': estate[0],
            'CategoryId': estate[1],
            'StreetId': estate[2],
            'UserId': estate[3],
            'Address': estate[4],
            'Title': estate[5],
            'Description': estate[6],
            'TypePost': estate[7],
            'Size': estate[8],
            'Price': estate[9],
            'Unit': estate[10],
            'Direction': estate[11],
            'BalconyDirection': estate[12],
            'FurnishingSell': estate[13],
            'Rooms': estate[14],
            'Toilets': estate[15],
            'Floors': estate[16],
            'Type': estate[17],
            'PropertyStatus': estate[18],
            'PropertyLegalDocument': estate[19],
            'Characteristics': estate[20],
            'Urgent': estate[21],
            'Images': estate[22],
        } for estate in real_estates
    ]

    # Return results as JSON
    return jsonify(real_estates_list)

@app.route('/trainHousePredictModel', methods=['POST'])
def trainHousePredictModel():
    try:
        # 1. Đọc dữ liệu từ CSV và chia thành X và y
        df = pd.read_csv('houseDataset.csv')

        df['DistrictLevel'] = df['DistrictId'].apply(map_district_level)
        df['DistrictLevel'] = df['DistrictLevel'].fillna(0).astype(int)

        df['StreetLevel'] = df['StreetId'].apply(map_street_level)
        df['StreetLevel'] = df['StreetLevel'].fillna(0).astype(int)

        X = df[['DistrictLevel', 'StreetLevel', 'Size', 'Rooms', 'Toilets', 'Floors', 'Type', 'FurnishingSell', 'Urgent', 'Characteristics']]
        y = df['Price']

        # 2. Xác định các cột phân loại và các cột số
        categorical_features = ['Type', 'FurnishingSell', 'Characteristics', 'Urgent']
        numerical_features = ['DistrictLevel', 'StreetLevel', 'Size', 'Rooms', 'Toilets', 'Floors']

        # 3. Thiết lập bộ tiền xử lý với handle_unknown='ignore'
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        # 4. Tạo pipeline bao gồm tiền xử lý và mô hình Linear Regression
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])

        # 5. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 6. Huấn luyện mô hình
        pipeline.fit(X_train, y_train)

        # 7. Lưu mô hình đã huấn luyện
        joblib.dump(pipeline, 'house_predict_model.pkl')

        # Return success message as JSON response
        return jsonify(dict(message='Model saved successfully!')), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
@app.route('/trainApartmentPredictModel', methods=['POST'])
def trainApartmentPredictModel():
    try:
        # 1. Đọc dữ liệu từ CSV và chia thành X và y
        df = pd.read_csv('apartmentDataset.csv')
        
        df['DistrictLevel'] = df['DistrictId'].apply(map_district_level)
        df['DistrictLevel'] = df['DistrictLevel'].fillna(0).astype(int)

        df['StreetLevel'] = df['StreetId'].apply(map_street_level)
        df['StreetLevel'] = df['StreetLevel'].fillna(0).astype(int)

        X = df[['DistrictLevel', 'StreetLevel', 'Size', 'Rooms', 'Toilets', 'Type', 'FurnishingSell', 'Urgent']]
        y = df['Price']

        # 2. Xác định các cột phân loại và các cột số
        categorical_features = ['DistrictLevel', 'StreetLevel','Type', 'FurnishingSell', 'Urgent']
        numerical_features = ['Size', 'Rooms', 'Toilets']

        # 3. Thiết lập bộ tiền xử lý với handle_unknown='ignore'
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),       # Chuẩn hóa các cột số
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-Hot Encoding cho các cột phân loại
            ]
        )

        # 4. Tạo pipeline bao gồm tiền xử lý và mô hình Linear Regression
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])

        # 5. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 6. Huấn luyện mô hình
        pipeline.fit(X_train, y_train)

        # 7. Lưu mô hình đã huấn luyện
        joblib.dump(pipeline, 'apartment_predict_model.pkl')

        # Return the predicted price as a JSON response
        return jsonify({
            'message': 'Save model success!'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/trainLandPredictModel', methods=['POST'])
def trainLandPredictModel():
    try:
        # 1. Đọc dữ liệu từ CSV và chia thành X và y
        df = pd.read_csv('landDataset.csv')
        
        df['DistrictLevel'] = df['DistrictId'].apply(map_district_level)
        df['DistrictLevel'] = df['DistrictLevel'].fillna(0).astype(int)

        df['StreetLevel'] = df['StreetId'].apply(map_street_level)
        df['StreetLevel'] = df['StreetLevel'].fillna(0).astype(int)

        X = df[['DistrictLevel', 'StreetLevel', 'Size', 'Type', 'Urgent', 'Characteristics']]
        y = df['Price']

        # 2. Xác định các cột phân loại và các cột số
        categorical_features = ['DistrictLevel', 'StreetLevel', 'Type', 'Characteristics', 'Urgent']
        numerical_features = ['Size']

        # 3. Thiết lập bộ tiền xử lý với handle_unknown='ignore'
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),       # Chuẩn hóa các cột số
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-Hot Encoding cho các cột phân loại
            ]
        )

        # 4. Tạo pipeline bao gồm tiền xử lý và mô hình Linear Regression
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])

        # 5. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 6. Huấn luyện mô hình
        pipeline.fit(X_train, y_train)

        # 7. Lưu mô hình đã huấn luyện
        joblib.dump(pipeline, 'land_predict_model.pkl')

        # Return the predicted price as a JSON response
        return jsonify({
            'message': 'Save model success!'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/trainCommercialPredictModel', methods=['POST'])
def trainCommercialPredictModel():
    try:
        # 1. Đọc dữ liệu từ CSV và chia thành X và y
        df = pd.read_csv('commercialDataset.csv')
        
        df['DistrictLevel'] = df['DistrictId'].apply(map_district_level)
        df['DistrictLevel'] = df['DistrictLevel'].fillna(0).astype(int)

        df['StreetLevel'] = df['StreetId'].apply(map_street_level)
        df['StreetLevel'] = df['StreetLevel'].fillna(0).astype(int)
        
        X = df[['DistrictLevel', 'StreetLevel', 'Size', 'Type', 'Urgent', 'FurnishingSell']]
        y = df['Price']

        # 2. Xác định các cột phân loại và các cột số
        categorical_features = ['Type', 'FurnishingSell', 'Urgent']
        numerical_features = ['DistrictLevel', 'StreetLevel', 'Size']

        # 3. Thiết lập bộ tiền xử lý với handle_unknown='ignore'
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),       # Chuẩn hóa các cột số
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-Hot Encoding cho các cột phân loại
            ]
        )

        # 4. Tạo pipeline bao gồm tiền xử lý và mô hình Linear Regression
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])

        # 5. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 6. Huấn luyện mô hình
        pipeline.fit(X_train, y_train)

        # 7. Lưu mô hình đã huấn luyện
        joblib.dump(pipeline, 'commercial_predict_model.pkl')

        # Return the predicted price as a JSON response
        return jsonify({
            'message': 'Save model success!'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/housePredict', methods=['POST'])
def housePredict():
    try:
        data = request.get_json()

        # Extract the input features from the request
        streetId = data.get('streetId')
        streetId_numeric = int(streetId) if streetId is not None else 0
        districtId = data.get('districtId')
        districtId_numeric = int(districtId) if districtId is not None else 0
        size = data.get('size')
        rooms = data.get('rooms')
        toilets = data.get('toilets')
        floors = data.get('floors')
        type = data.get('type')
        furnishingSell = data.get('furnishingSell')
        urgent = data.get('urgent')
        urgent_numeric = int(urgent) if urgent is not None else 0
        characteristics = data.get('characteristics')

        districtLevel = map_district_level(districtId_numeric)
        streetLevel = map_street_level(streetId_numeric)

        new_house = pd.DataFrame({
            'DistrictLevel': [districtLevel],
            'StreetLevel': [streetLevel],
            'Size': [size],
            'Rooms': [rooms],
            'Toilets': [toilets],
            'Floors': [floors],
            'Type': [type],
            'FurnishingSell': [furnishingSell],
            'Urgent': [urgent_numeric],
            'Characteristics': [characteristics],
        })

        # Tải lại mô hình đã lưu
        loaded_model = joblib.load('house_predict_model.pkl')

        # Dự đoán giá cho ngôi nhà mới
        predicted_price = loaded_model.predict(new_house)[0]

        # Đảm bảo giá trị là số dương và làm tròn đến hàng triệu
        predicted_price = abs(predicted_price)  # Chuyển thành số dương nếu cần
        rounded_price = round(predicted_price, -6)  # Làm tròn đến hàng triệu

        # Return the predicted price as a JSON response
        return jsonify({
            'predicted_price': f"{rounded_price:,.0f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/apartmentPredict', methods=['POST'])
def apartmentPredict():
    try:
        data = request.get_json()

        # Extract the input features from the request
        streetId = data.get('streetId')
        streetId_numeric = int(streetId) if streetId is not None else 0
        districtId = data.get('districtId')
        districtId_numeric = int(districtId) if districtId is not None else 0
        size = data.get('size')
        rooms = data.get('rooms')
        toilets = data.get('toilets')
        type = data.get('type')
        furnishingSell = data.get('furnishingSell')
        urgent = data.get('urgent')
        urgent_numeric = int(urgent) if urgent is not None else 0

        districtLevel = map_district_level(districtId_numeric)
        streetLevel = map_street_level(streetId_numeric)

        new_apartment = pd.DataFrame({
            'DistrictLevel': [districtLevel],
            'StreetLevel': [streetLevel],
            'Size': [size],
            'Rooms': [rooms],
            'Toilets': [toilets],
            'Type': [type],
            'FurnishingSell': [furnishingSell],
            'Urgent': [urgent_numeric],  
        })

        # Tải lại mô hình đã lưu
        loaded_model = joblib.load('apartment_predict_model.pkl')

        # Dự đoán giá cho ngôi nhà mới
        predicted_price = loaded_model.predict(new_apartment)[0]

        # Đảm bảo giá trị là số dương và làm tròn đến hàng triệu
        predicted_price = abs(predicted_price)  # Chuyển thành số dương nếu cần
        rounded_price = round(predicted_price, -6)  # Làm tròn đến hàng triệu

        # Return the predicted price as a JSON response
        return jsonify({
            'predicted_price': f"{rounded_price:,.0f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/landPredict', methods=['POST'])
def landPredict():
    try:
        data = request.get_json()

        # Extract the input features from the request
        streetId = data.get('streetId')
        streetId_numeric = int(streetId) if streetId is not None else 0
        districtId = data.get('districtId')
        districtId_numeric = int(districtId) if districtId is not None else 0
        size = data.get('size')
        type = data.get('type')
        characteristics = data.get('characteristics')
        urgent = data.get('urgent')
        urgent_numeric = int(urgent) if urgent is not None else 0
        
        districtLevel = map_district_level(districtId_numeric)
        streetLevel = map_street_level(streetId_numeric)

        new_land = pd.DataFrame({
            'DistrictLevel': [districtLevel],
            'StreetLevel': [streetLevel],
            'Size': [size],
            'Type': [type],
            'Urgent': [urgent_numeric],  
            'Characteristics': [characteristics],
        })

        # 8. Tải lại mô hình đã lưu
        loaded_model = joblib.load('land_predict_model.pkl')

        # 9. Dự đoán giá cho ngôi nhà mới
        predicted_price = loaded_model.predict(new_land)[0]

        # Đảm bảo giá trị là số dương và làm tròn đến hàng triệu
        predicted_price = abs(predicted_price)  # Chuyển thành số dương nếu cần
        rounded_price = round(predicted_price, -6)  # Làm tròn đến hàng triệu

        # Return the predicted price as a JSON response
        return jsonify({
            'predicted_price': f"{rounded_price:,.0f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/commercialPredict', methods=['POST'])
def commercialPredict():
    try:
        data = request.get_json()

        # Extract the input features from the request
        streetId = data.get('streetId')
        streetId_numeric = int(streetId) if streetId is not None else 0
        districtId = data.get('districtId')
        districtId_numeric = int(districtId) if districtId is not None else 0
        size = data.get('size')
        type = data.get('type')
        furnishingSell = data.get('furnishingSell')
        urgent = data.get('urgent')
        urgent_numeric = int(urgent) if urgent is not None else 0
        
        districtLevel = map_district_level(districtId_numeric)
        streetLevel = map_street_level(streetId_numeric)

        new_commercial = pd.DataFrame({
            'DistrictLevel': [districtLevel],
            'StreetLevel': [streetLevel],
            'Size': [size],
            'Type': [type],
            'FurnishingSell': [furnishingSell],
            'Urgent': [urgent_numeric],  
        })

        # Tải lại mô hình đã lưu
        loaded_model = joblib.load('commercial_predict_model.pkl')

        # Dự đoán giá cho ngôi nhà mới
        predicted_price = loaded_model.predict(new_commercial)[0]

        # Đảm bảo giá trị là số dương và làm tròn đến hàng triệu
        predicted_price = abs(predicted_price)  # Chuyển thành số dương nếu cần
        rounded_price = round(predicted_price, -6)  # Làm tròn đến hàng triệu

        # Return the predicted price as a JSON response
        return jsonify({
            'predicted_price': f"{rounded_price:,.0f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
