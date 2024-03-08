# Relaxation-of-rods-in-electrical-cell
 ode for analysing the optical retardation relaxation of nanorods in  a opto electrical cell
This piece of code in python3 was developed in order to automatize some data analysis to  make it quicker. 
The experiment was birefringence relaxation in a optoelectrical cell (read following if interested):

	Birefringence is a phenomenon that consists in the change in the optical index of some material. It can manifest in materials with some structural anisotropy such as fibers, polymers or elongated nanoparticles as in this case
	In this case in fact we are dealing with LaPO4:Eu nanorods. These are dissolved in a media  (water or ethylene glycol) and a drop of this slutionn is placed intno an alectro  optical cell. The ladder consists simply of a glass substrate with some deposited gold electrodes (2 parallel stripes if gold with a gap in between : V-----| |-----0).
	By applying an oscillating potential on the electrodes the rods tend to allign in the field direction. Allignment causes biefringence. 
	We shine a liht through the cell and into a camera. By using polarisers before and after the cell we are able to see a change in light intensity with  thwe birefringence. 
	To make it short : Field ON -> bright, field  OFF -> dark.
	The aim of these experiments is to measure the relaxation time of the rods when the  field is suddently switched off.
	An **important note** in order to understand the analysiis. We want to analyse the so called OPTICAL RETARDATION (OR) which is proportional to the birefringence. To obtain this we need to know what is the intensity of the light when the above mentioned polarizers are parallel. This is called I_0 in the code. Back to the important stuff.

The collected data consists of an tif image (img0.tif) (as mentioned in the  note above) and one or more multipage tif called relax*.tif . Note that it is easy to change the name formats of the images in the  first lines of code.
The intensity of all of the images is automatically read from the images by seeing the pixel grey scale value on a stripe in the  middle of the  image. This was set to be the case for my measurements but can be rather easily changed too.
After connstructing a list with the intensities and one with the time (frames/fps) the code finds automatically the beginning of the decay and fits  with a stretched exponential of the form I = exp[-(6Dt)^alpha] to find both D (Diffusion parameter) and alpha.  This is  done for every file called rela*.tif and an average  is given.

The only parameters that need to be manually set are :
- path of the folder containing relax (same folder)
- path of the img0.tif image
- name structure of rela file if you want to change it
- total length of the videos. The videos should be taken of the same length because  the way to calculate the fps is frames/lenngth of video. 

Some parameters that might need adjustments but not necessaril :
-fit_len is the length of the fit in frames. The code will fit the stretched exponential from the detected  drop on, for the  given number of frames (or points).
-The percentage value in the function for finnding the drop (see diectly in the code).
-heigth_delimeters determine where the code measures the intensity of the pixel.

Options to visualize different graphs are given inn form of boolean right after specifying the paths of the tif. 
-show_patch is for checking that a correct portionn of the images is being measured: it will show img0 with thw area patch drawed on 
-visualize_derivative is to shows the derivative of the smothened data multiplied by 5. this is used to find the drop. It should ideally show a flat line with one peak where the drop happens. In reality there are many peaks because of the noise in the data. 
-visualise_drop is to visualize what portion of the data is identified as the drop (just the beginnng of it is meaningful).
- visualize_dataset shows thr dataset. It can be somewhat redondant with the previous oprion. It is usefl if we want to make sure the data is being collected properly or that the videeos arre good enough. 
-visualize_fit showws thee  fit on the drop part of the data. 



