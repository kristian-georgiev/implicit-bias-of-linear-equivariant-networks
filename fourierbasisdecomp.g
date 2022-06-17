LoadPackage("RepnDecomp");

#This is a helper function that returns a vector reading out
#the elements of a matrix from top-->bottom and left-->right
#e.g.  [[1,2],[3,4]] --> [1,3,2,4]
FlattenMx := function(M)
	local v, col;	
	v := [];
	for col in TransposedMat(M) do
		Append(v, col);
	od;
	return v;
end;

ComputeFourierMx := function(G)
	local F, irreps, rho, ur, g, canonical_block, rows_transpose, im, scale;
	
	#The first row contains a list of group elements as GAP represents them
	#This first row is necessary because we do not have a canonical ordering of the group	
	F := [[],[]];
	for g in G do
		Add(F[1], g);
	od;
	
	irreps := IrreducibleRepresentations(G);
	for rho in irreps do
		#scale := Sqrt(Length(ImagesRepresentative(canonical_block, G.1))/Size(G));
		#The above expression gives the correct scaling factor for the row. However,
		#we return an unscaled matrix and apply scaling in Sage for precision reasons
		scale := 1;

		#if an irrep is not unitary, unitarize it.
		Print("rho = ", rho, " is unitary? ", IsUnitaryRepresentation(rho), "\n\n");
		ur := UnitaryRepresentation(rho).unitary_rep;

		#Include a row with the dimensions of the irreps
		Add(F[2], Length(ImagesRepresentative(ur, G.1)));
		
		#Build up the Fourier basis matrix via little column slices 
		rows_transpose := [];
		for g in G do
			im := ImagesRepresentative(ur, g);
			Add(rows_transpose, scale*FlattenMx(im));
		od;

	Append(F, TransposedMat(rows_transpose));
	od;
	return F;
end;

#Below is sample execution of the above functions
G := DihedralGroup(8);
FF := ComputeFourierMx(G);
output := OutputTextFile("data/unscaled_bases/D8", false);
SetPrintFormattingStatus(output, false);
PrintTo(output, FF);
