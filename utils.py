

def create_experiment_directory(self):
        """
        Crea un nuevo directorio para el experimento actual.
        """
        base_dir = 'experiments'
        os.makedirs(base_dir, exist_ok=True)
        
        # Encontrar el siguiente número de ejecución
        existing_runs = [d for d in os.listdir(base_dir) if d.startswith('run')]
        if not existing_runs:
            next_run = 1
        else:
            run_numbers = [int(run.replace('run', '')) for run in existing_runs]
            next_run = max(run_numbers) + 1
            
        # Crear directorio para esta ejecución
        run_dir = os.path.join(base_dir, f'run{next_run}')
        os.makedirs(run_dir)
        
        # Crear subdirectorios para diferentes tipos de archivos
        os.makedirs(os.path.join(run_dir, 'plots'))
        os.makedirs(os.path.join(run_dir, 'metrics'))
        os.makedirs(os.path.join(run_dir, 'model'))
        
        return run_dir