import PySimpleGUI as sg

def images_explorer():
    images = []

    # UI
    layout = [  [sg.LBox([], size = (70, 20), key = '-FILESLB-',)],
                [sg.Input(visible = False, enable_events = True, key = '-IN-'), sg.FilesBrowse('Explorar')],
                [sg.Button('Guardar', key = '-BTN-SAVE-')]
            ]

    # Window object
    window = sg.Window('App', layout)

    # Infinite Event Loop
    while True:             
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        
        if event == '-IN-':
            images += values['-IN-'].split(';')
            window['-FILESLB-'].Update(images)
        
        elif event == '-BTN-SAVE-':
            window.close();    
            return images