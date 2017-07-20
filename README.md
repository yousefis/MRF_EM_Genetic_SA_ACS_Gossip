# MRF_EM_Genetic_SA_ACS_Gossip
# Latest version: 2017-07-7
This is a Segmentation method based on MRF as proposed by Sahar Yousefi et al. 

Developer: 

	Sahar Yousefi, syousefi(-at-)ce.sharif.edu	

Functions:

	In this project the below algorithms for brain MRI segmentation are implemented:
		1- Expectation Maximization
		2- Traditional MRF models contain:
			MMD_Metropolis
			ICM
			Gibbs
		3- MetropolisGenetic
		4- MetropolisACS
	For more information you can study the papers which are mentioned in the Licence section. 
	
	MRF_EM_Genetic_SA_ACS_Gossip(imanm,man_seg,k)
	Input:
		imanm: gray color image name
		k: Number of classes 
	Output:
		mask: clasification image mask
		mu: vector of class means 
		v: vector of class variances
		p: vector of class proportions  
		   
Licence:

	You are welcome to use this code in your own work. 
	If you use this code, please cite one (or more) of these papers:
		1- Yousefi, Sahar, Reza Azmi, and Morteza Zahedi. "Brain tissue segmentation in MR images based on a hybrid of MRF and social algorithms." Medical image analysis 16.4 (2012): 840-848.
		2- Yousefi, Sahar, Morteza Zahedi, and Reza Azmi. "3D MRI brain segmentation based on MRF and hybrid of SA and IGA." Biomedical Engineering (ICBME), 2010 17th Iranian Conference of. IEEE, 2010.
		3- Ahmadvand, Ali, Sahar Yousefi, and M. T. Manzuri Shalmani. "A novel Markov random field model based on region adjacency graph for T1 magnetic resonance imaging brain segmentation." International Journal of Imaging Systems and Technology 27.1 (2017): 78-88.
	   
	   
QUESTIONS:

	May you have any question, please contact 

		syousefi(-at-)ce.sharif.edu

	All the best in your endeavours. :)
