import matplotlib.pyplot as plt
import numpy as np

from record import create_record 
import pandas as pd
import hyperparameters as hp


metadata_df = pd.read_csv(hp.METADATA_PATH)

num_patients = len(metadata_df["patient_id"].unique())
print(f"Number of patients: {num_patients}")
num_records = len(metadata_df["record_id"].unique())
print(f"Number of records: {num_records}")


record = create_record("record_026", metadata_df, hp.RECORDS_PATH)
record.load_ecg()
print(record.ecg[0].shape)
#Plot of the ECG record_026
plt.figure(1,figsize=(12, 7))
plt.rcParams.update({'font.size': 16})
plt.subplot(2, 1, 1)
plt.plot(record.ecg[0][:, 0],'k')
plt.title('record 026')
plt.ylabel("Lead I (mV)")
plt.xlabel("Sample index")
plt.subplot(2, 1, 2)
plt.plot(record.ecg[0][4684080-1000:4684080+1000, 0],'k')
plt.title('record 026 partial')
plt.ylabel("Lead I (mV)")
plt.xlabel("Sample index")
xticks_labels = [str(i) for i in range(4684080 - 1000 , 4684080 + 1000, 200)]
plt.xticks(range(0, len(record.ecg[0][4684080-1000:4684080+1000, 0]), 200), xticks_labels)
plt.xticks(rotation=45)
plt.tight_layout()  
plt.show()

#Plot record_026 at the start of AF period
plt.figure(2,figsize=(12,7))
plt.subplot(2, 1, 1)
plt.plot(record.ecg[0][4684080-1000:4684080+1000, 0],'k')
plt.title('record 026 partial')
plt.ylabel("Lead I (mV)")
plt.xlabel("Sample index")
xticks_labels = [str(i) for i in range(4684080 - 1000 , 4684080 + 1000, 200)]
plt.xticks(range(0, len(record.ecg[0][4684080-1000:4684080+1000, 0]), 200), xticks_labels)
plt.xticks(rotation=45)
shift = 5 * 200
#print(record.ecg_labels_df.iloc[()])
event_index = record.ecg_labels_df.iloc[0].start_qrs_index
af_data = record.ecg[0][event_index - shift:event_index + shift, 0]
af_before = np.copy(af_data)
af_before[shift:] = np.nan
af_after = np.copy(af_data)
af_after[:shift] = np.nan
plt.subplot(2, 1, 2)
plt.plot(af_before, color="green")
plt.plot(af_after, color="red")
plt.ylabel("Lead I (mV)")
plt.xlabel("Sample index")
plt.axvline(x=shift, color="black", linestyle="--")
xticks_labels = [str(i) for i in range(event_index - shift, event_index + shift, 200)]
plt.xticks(range(0, len(af_data), 200), xticks_labels)
plt.xticks(rotation=45)
plt.title("Start of AF period")
plt.tight_layout()
plt.show()


plt.figure(3,figsize=(12,7))
plt.subplot(2, 1, 1)
plt.plot(record.ecg[0][0:8192, 0],'k')
plt.ylabel("Lead I (mV)")
plt.xlabel("Sample index")
plt.title('standard ECG')
plt.subplot(2, 1, 2)
plt.plot(record.ecg[0][8011776:8019968, 0],'k')
plt.ylabel("Lead I (mV)")
plt.xlabel("Sample index")
xticks_labels = [str(i) for i in range(8011776, 8019968, 2000)]
plt.xticks(range(0, len(record.ecg[0][8011776:8019968, 0]), 2000), xticks_labels)
plt.xticks(rotation=45)
plt.title('AF ECG')
plt.tight_layout()
plt.show()