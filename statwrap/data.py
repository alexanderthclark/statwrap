'''
Data sets and random data functions.
'''
import pandas as pd

# Inspired by Galenson's Old Masters and Young Geniuses
# These ages and dates are all approximate.
picasso = {
    "artist": ["Pablo Picasso"] * 50,
    "painting": [
        "The First Communion", "Science and Charity", "Self-portrait",
        "The Death of Casagemas", "Evocation (The Burial of Casagemas)",
        "The Old Guitarist", "La Vie", "Boy Leading a Horse", "Les Demoiselles d'Avignon",
        "Three Women", "The Reservoir, Horta de Ebro", "Portrait of Ambroise Vollard",
        "Girl with a Mandolin", "The Poet", "Guitar, Sheet Music, and Wine Glass",
        "Still Life with Chair Caning", "Three Musicians", "The Three Dancers", 
        "The Crucifixion", "Girl before a Mirror", "Le Rêve", "Guernica", 
        "The Weeping Woman", "Night Fishing at Antibes", "The Charnel House", 
        "Massacre in Korea", "Don Quixote", "Les Femmes d'Alger", "Jacqueline with Flowers", 
        "The Rape of the Sabine Women", "Man and Woman", "Seated Woman", 
        "Bull's Head", "The Kitchen", "Woman with a Hat", "The Kiss", 
        "The Dance", "Bust of a Woman", "Dove of Peace", "Large Bather with a Book", 
        "Vollard Suite", "Bather with Beach Ball", "Harlequin", "Portrait of Olga in an Armchair", 
        "Reading the Letter", "Minotauromachy", "Mother and Child by the Sea", 
        "Man with a Pipe", "Sleeping Peasants", "The Matador"
    ],
    "date_of_completion": [
        1896, 1897, 1901, 1901, 1901, 1903, 1903, 1906, 1907, 1908,
        1909, 1910, 1910, 1912, 1912, 1912, 1921, 1925, 1930, 1932,
        1932, 1937, 1937, 1939, 1945, 1951, 1955, 1955, 1956, 1963,
        1969, 1930, 1942, 1948, 1962, 1969, 1951, 1944, 1949, 1937,
        1937, 1932, 1915, 1918, 1921, 1935, 1902, 1911, 1919, 1970
    ],
    "age": [
        15, 16, 20, 20, 20, 22, 22, 25, 26, 27,
        28, 29, 29, 31, 31, 31, 40, 44, 49, 51,
        51, 56, 56, 58, 64, 70, 74, 74, 75, 82,
        88, 49, 61, 67, 81, 88, 70, 63, 68, 56,
        56, 51, 34, 37, 40, 54, 21, 30, 38, 89
    ]
}

cezanne = {
    "artist": ["Paul Cezanne"] * 50,
    "painting": [
        "The Murder", "Modern Olympia", "The Bather", "Boy in a Red Waistcoat", "Mont Sainte-Victoire",
        "The Large Bathers", "Still Life with Apples", "The Basket of Apples", "The Card Players", "The Bay of Marseille",
        "Houses in Provence", "L'Estaque", "The Sea at L'Estaque", "Still Life with Curtain", "Portrait of Gustave Geffroy",
        "Chateau Noir", "Pyramid of Skulls", "Mont Sainte-Victoire Seen from Bellevue", "Mardi Gras", "Madame Cézanne in a Yellow Chair",
        "Still Life with Cherub", "Mont Sainte-Victoire and the Viaduct", "Portrait of Ambroise Vollard", "Jas de Bouffan", "Lac d'Annecy",
        "Sugar Bowl, Pears and Tablecloth", "View of the Domaine Saint-Joseph", "The Brook", "Madame Cézanne with Loosened Hair", "Portrait of Victor Chocquet",
        "Mont Sainte-Victoire with Large Pine", "Still Life with Plaster Cupid", "Portrait of Louis Guillaume", "Chestnut Trees and Farm of the Jas de Bouffan", "Still Life, Drapery, Pitcher, and Fruit Bowl",
        "Turning Road at Montgeroult", "Mont Sainte-Victoire and Château Noir", "The Gardener Vallier", "Man with a Pipe", "Still Life with Teapot",
        "Bather with Outstretched Arms", "The Forest", "Mont Sainte-Victoire Seen from the Bibémus Quarry", "Apples and Oranges", "The Avenue at the Jas de Bouffan",
        "The Well at Jas de Bouffan", "The Lac d’Annecy", "Still Life with Water Jug", "The Grounds of the Château Noir", "Mount Sainte-Victoire"
    ],
    "date_of_completion": [
        1868, 1874, 1887, 1890, 1887, 1905, 1890, 1894, 1892, 1885,
        1883, 1885, 1876, 1895, 1896, 1904, 1900, 1892, 1888, 1890,
        1895, 1885, 1899, 1887, 1896, 1894, 1887, 1900, 1890, 1877,
        1887, 1895, 1882, 1885, 1894, 1898, 1906, 1906, 1892, 1906,
        1899, 1892, 1897, 1899, 1878, 1885, 1896, 1893, 1904, 1904
    ],
    "age": [
        29, 35, 48, 51, 48, 63, 51, 55, 53, 46,
        44, 46, 37, 56, 57, 65, 61, 53, 49, 51,
        56, 46, 60, 48, 57, 55, 48, 61, 51, 38,
        48, 56, 43, 46, 55, 59, 67, 67, 53, 67,
        60, 53, 58, 60, 39, 46, 57, 54, 65, 65
    ]
}

paintings = pd.concat([pd.DataFrame(picasso), pd.DataFrame(cezanne)])
