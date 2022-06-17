a := CyclicGroup(10);
b := CyclicGroup(10);
c := CyclicGroup(2);
G := DirectProduct(a, b, c);
output := OutputTextFile("data/cayley/C10C10C2_Cayley", false);

for g in G do
    for h in G do
        PrintTo(output, g * h);
        PrintTo(output, ", ");
    od;
od;