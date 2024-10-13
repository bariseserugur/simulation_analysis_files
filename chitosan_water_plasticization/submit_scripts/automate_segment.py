import os

submitf = open('run_segment.submit').readlines()

for conc in [0,1,5,10,15,20,30,40]:
    silf = open('automatic_run_segment.submit','w')
    for i in submitf:
        if 'cd ' in i:
            i = i.replace('aa','{}'.format(conc))
        silf.write(i)
    silf.close()
    os.system('sbatch automatic_run_segment.submit')
 
