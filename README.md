# FALCON

## Flow Analysis Tools for Large-Scale Complex Networks.

FALCON is a tool to analyze the flow structure of a directed network.
There are two main tools, one to perform Helmholtz-Hodge decomposition for the network.
This allows you to decompose a directed network into Gradient flow and Loop flow components and determine potential of each node. This means its position in the flow network.
The other is a tool to calculate 3D coordinates for network visualization by using a spring-charge model of molecular dynamics(MD).
The tool is customized to run MD with the Helmholtz-Hodge potential fixed on the Z axis, allowing visualizations that reflect the flow structure of the network.


## Usage

- Download files hh_dcmp.cpp for Helmholtz-Hodge decomposition and MD_z_const.cpp for MD.
- These programs are use the C++ Eigen library.
- To compile from your shell or cmd command: `g++ hh_dcmp.ccp -I Eigen -o hh_dcmp.exe`
- You can use the -fopenmp option if you want parallelization.

- hh_dcmp.exe accepts an input file in edgelist format.
2 or 3 columns edgelist data are acceptable.
1st and 2nd colmn are node id (integer) and  3rd colmn is weight of link. The orientation of the link is from the first column to the second column. For two-column data, all weights are set to 1.

> edgelist.dat  
> 1 6 1.041  
> 2 7 0.590  
> 3 8 0.274  
> 4 9 0.225  
> 5 10 0.935
>


`> hh_dcmp.exe edgelist.dat`

- hh_dcmp.exe offers four outputs.  
    edgelist_log.dat  
    edgelist_loop_flow.dat  
    edgelist_potential.dat  
    edgelist_potential_flow.dat  

- MD_z_const.exe accepts edgelist and initial coordinate data as input file.

> init_coord.dat  
> 1 -1.455  0.687 -0.783  
> 2 -0.180  0.743  0.691  
> 3  0.831  0.359  1.212  
> 4  1.605  0.169 -1.293  
> 5 -0.387  0.389  1.274  


`> MD_z_const.exe edgelist.dat init_coord.dat`

- The z coordinate is not updated from the initial value, so using the Helmholtz-Hodge potential as the z coordinate gives results along the network flow structure.

## Examples

Initial coordinate.
<img src=before_color.png width=400>

After applying MD simulation.
<img src=after_color.png width=400>

Visualization of the blog data with bow-tie division colors.
<img src=blog_bowtie.png width=400>

## Publications based on FALCON

Kichikawa, Y., Iino, T., Iyetomi, H. et al. J Comput Soc Sc (2019). https://doi.org/10.1007/s42001-019-00031-1  
Kichikawa, Y., Iyetomi, H., Iino, T. et al. Community structure based on circular flow in a large-scale transaction network. Appl Netw Sci 4, 92 (2019). https://doi.org/10.1007/s41109-019-0202-8  
KICHIKAWA, Yuichi; IYETOMI, Hiroshi; IKEDA, Yuichi. Who Possesses Whom in Terms of the Global Ownership Network. In: Big Data Analysis on Global Community Formation and Isolation: Sustainability and Flow of Commodities, Money, and Humans. Singapore: Springer Singapore, 2021. p. 143-190. https://doi.org/10.1007/978-981-15-4944-1_6
