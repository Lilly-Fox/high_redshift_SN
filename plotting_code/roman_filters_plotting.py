import matplotlib.pyplot as plt
from astropy import units as u
from astroquery.svo_fps import SvoFps

data_list = []

roman_filters = ['Roman/WFI.F062', 'Roman/WFI.F087', 'Roman/WFI.F106', 'Roman/WFI.F129', 'Roman/WFI.F158', 'Roman/WFI.F184', 'Roman/WFI.F213'] #'Roman/WFI.F146',
for filter in roman_filters:
    data = SvoFps.get_transmission_data(filter)
    data_list.append((data['Wavelength'], data['Transmission']))

plt.figure(figsize=(10, 6))
for x, y in data_list:
    plt.plot(x, y)
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux')
plt.grid(True)
plt.tight_layout()
plt.savefig(r"C:\Users\lmf53\OneDrive - University of Canterbury\CAS\graphs_and_figures\initial_roman_filters.png")
plt.show()