for T in $(seq 295 5 360); do
  DIR="${T}K"
  echo "==> Preparando directorio ${DIR}"
  
  # 1. Crear directorio (si no existe)
  mkdir -p "${DIR}"
  
  # 2. Copiar ficheros plantilla
  cp job.sh simulate.py "${DIR}/"
  
  # 3. Sustituir la palabra TEMPERATURE por el valor numérico
  sed -i "s/TEMPERATURE/${T}/g" "${DIR}/job.sh" "${DIR}/simulate.py"
  
  # 4. Enviar el job
  (
    cd "${DIR}"
    echo "    -> Enviando sbatch job.sh"
    sbatch job.sh
  )
done

echo "Todos los jobs han sido enviados."
