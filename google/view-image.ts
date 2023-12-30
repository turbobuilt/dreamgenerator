import * as fs from 'fs';
import sharp from 'sharp';

async function main() {
    try {
        const data = await fs.promises.readFile('/Users/dev/Documents/prg/dreamgenerator.ai/google/output.json', 'utf-8');
        const jsonData = JSON.parse(data);
        const base64Encoded = jsonData.predictions[0].bytesBase64Encoded;

        
        const directoryPath = './images';
        const files = await fs.promises.readdir(directoryPath);
        let highestNumber = 0;

        // Find the highest numbered image
        const regex = /(\d+)\.avif/;
        files.forEach((file) => {
            const match = file.match(regex);
            if (match) {
                const number = parseInt(match[1]);
                if (number > highestNumber) {
                    highestNumber = number;
                }
            }
        });

        const buffer = Buffer.from(base64Encoded, 'base64');
        await sharp(buffer)
            .toFile(`/Users/dev/Documents/prg/dreamgenerator.ai/google/images/${highestNumber}.avif`);
            
        console.log('PNG file saved successfully!');
    } catch (error) {
        console.error('Error:', error);
    }
}

main();
