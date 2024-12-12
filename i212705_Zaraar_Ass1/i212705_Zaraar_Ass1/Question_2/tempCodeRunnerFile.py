# Generate Unique Word Dictionary for Word-Vectors
DICT=helper_functions.vector_dict(x,y)

num_x=helper_functions.convert_text_to_vectors(DICT,x)
num_y=helper_functions.convert_TOV_y(DICT,y)
pad_x=helper_functions.pad(num_x,DICT)

x_train, x_test, y_train, y_test = train_test_split(pad_x, num_y, test_size=0.2, shuffle=True, random_state=34)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=34)

train_dataset = CustomDataset(DICT,x_train,y_train)
val_dataset = CustomDataset(DICT,x_val,y_val)
# test_dataset = CustomDataset(DICT,x_test,y_test)

# # # # # Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) 
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) 

# # print('here')
# # model=LSTMModel(DICT,train_loader,test_loader)
MODEL=model.train_test(DICT,train_loader,val_loader)
